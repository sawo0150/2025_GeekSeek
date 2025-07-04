import numpy as np
import torch

import numpy.fft as fft
import h5py
import os
from tqdm import tqdm

class GCCCompressor:
    """
    MRI k-space 데이터에 대해 하이브리드 공간 GCC(Geometric-decomposition Coil Compression)를
    수행하는 클래스.

    이 클래스는 4D k-space 데이터(슬라이스, 코일, PE, RO)를 입력받아
    지정된 가상 코일 수로 압축하고, 그 결과를 HDF5 파일로 저장합니다.
    """
    def __init__(self, 
                 target_coils=3, 
                 num_calib_lines=48, 
                 sliding_window_size=5, 
                 compression_dim=1,
                 kspace_dataset_name='kspace'):
        """
        GCCCompressor 클래스의 생성자.

        Args:
            target_coils (int): 압축 후 목표 가상 코일의 수.
            num_calib_lines (int): k-space 중앙에서 추출할 보정 데이터 라인(PE)의 수.
            sliding_window_size (int): GCC 행렬 계산 시 사용할 슬라이딩 윈도우의 크기.
            compression_dim (int): 압축을 적용할 차원. GCC는 Readout 방향으로 적용되므로 1.
                                  (데이터 형태가 (PE, RO, Coils)일 때 기준)
            kspace_dataset_name (str): HDF5 파일 내의 k-space 데이터셋 이름.
        """
        self.target_coils = target_coils
        self.num_calib_lines = num_calib_lines
        self.sliding_window_size = sliding_window_size
        self.compression_dim = compression_dim
        self.kspace_dataset_name = kspace_dataset_name
        print("GCC Compressor가 다음 파라미터로 초기화되었습니다:")
        print(f"- Target Coils: {self.target_coils}")
        print(f"- Calibration Lines: {self.num_calib_lines}")
        print(f"- Sliding Window Size: {self.sliding_window_size}")

    def _calc_gcc_matrices(self, calib_data):
        """
        (비공개) 보정 데이터를 '하이브리드 공간'으로 변환 후, 각 위치에 대한 GCC 압축 행렬을 계산합니다.
        """
        if self.compression_dim == 1:
            # (PE, RO, Coils) -> (RO, PE, Coils)
            calib_data_transposed = calib_data.transpose(1, 0, 2)
        else:
            calib_data_transposed = calib_data
        
        # Readout 축(이제 첫 번째 축)을 따라 iFFT를 수행하여 하이브리드 공간으로 변환
        hybrid_calib_data = fft.ifft(fft.ifftshift(calib_data_transposed, axes=0), axis=0)
        
        n_readout, _, n_coils = hybrid_calib_data.shape
        gcc_matrices = np.zeros((n_readout, n_coils, n_coils), dtype=np.complex128)
        
        for i in tqdm(range(n_readout), desc="  Calculating GCC matrices", leave=False, ascii=True):
            start = max(0, i - self.sliding_window_size // 2)
            end = min(n_readout, i + self.sliding_window_size // 2 + 1)
            
            block = hybrid_calib_data[start:end, :, :]
            reshaped_block = block.reshape(-1, n_coils)
            _, _, vh = np.linalg.svd(reshaped_block, full_matrices=False)
            gcc_matrices[i, :, :] = vh.conj().T
            
        return gcc_matrices

    def _align_gcc_matrices(self, gcc_matrices):
        """
        (비공개) 계산된 GCC 행렬들을 정렬하여 부드럽게 변화하도록 만듭니다.
        """
        n_readout, n_coils, _ = gcc_matrices.shape
        cropped_matrices = gcc_matrices[:, :, :self.target_coils]
        aligned_matrices = np.zeros_like(cropped_matrices, dtype=np.complex128)
        aligned_matrices[0] = cropped_matrices[0]
        
        for i in range(1, n_readout):
            prev_aligned_v = aligned_matrices[i-1]
            current_v = cropped_matrices[i]
            correlation_matrix = prev_aligned_v.T.conj() @ current_v
            u_rot, _, vh_rot = np.linalg.svd(correlation_matrix, full_matrices=False)
            rotation_matrix = u_rot @ vh_rot
            aligned_matrices[i] = current_v @ rotation_matrix
            
        return aligned_matrices

    def _apply_compression(self, kspace_data, compression_matrices):
        """
        (비공개) 전체 k-space 데이터에 압축 행렬을 적용합니다.
        """
        if self.compression_dim == 1:
            # kspace(p,r,c)와 matrix(r,c,v)를 곱해 compressed_kspace(p,r,v) 생성
            return np.einsum('prc, rcv -> prv', kspace_data, compression_matrices)
        else:
            # kspace(p,r,c)와 matrix(p,c,v)를 곱해 compressed_kspace(p,r,v) 생성
            return np.einsum('prc, pcv -> prv', kspace_data, compression_matrices)

    def process_slice(self, kspace_slice_3d):
        """
        한 개의 3D k-space 슬라이스 데이터에 대해 GCC 압축을 수행합니다.
        [수정됨] 올바른 하이브리드 공간에서 압축을 적용합니다.

        Args:
            kspace_slice_3d (np.ndarray): (PE, RO, Coils) 형태의 3D k-space 데이터.

        Returns:
            tuple[np.ndarray, np.ndarray]: Unaligned 압축 결과, Aligned 압축 결과.
        """
        # 1. 보정 데이터 추출 (k-space 중앙) - 변경 없음
        n_pe, _, _ = kspace_slice_3d.shape
        start = n_pe // 2 - self.num_calib_lines // 2
        end = n_pe // 2 + self.num_calib_lines // 2
        calib_data = kspace_slice_3d[start:end, :, :]

        # 2. GCC 행렬 계산 - 변경 없음
        # 이 함수는 이미지 공간(x)에 대한 압축 행렬을 올바르게 계산합니다.
        gcc_matrices_full = self._calc_gcc_matrices(calib_data)

        # --- [핵심 수정 부분 시작] ---

        # 3. 전체 k-space 데이터를 하이브리드 공간(x, k_y)으로 변환합니다.
        # (PE, RO, Coils) -> (RO, PE, Coils)로 축 변경
        kspace_transposed = kspace_slice_3d.transpose(1, 0, 2)
        # RO(k_x) 방향으로 iFFT를 수행하여 x 차원으로 변환
        hybrid_space_full = fft.ifft(fft.ifftshift(kspace_transposed, axes=0), axis=0)

        # 4. 하이브리드 공간에서 압축 행렬을 적용합니다.
        # einsum('prc,rcv->prv')에 맞게 (PE,RO,Coils) 형태로 다시 축 변경
        # 여기서 PE는 k_y, RO는 x에 해당합니다.
        hybrid_space_for_einsum = hybrid_space_full.transpose(1, 0, 2)

        # 4-1. Unaligned 압축 적용
        gcc_matrices_unaligned = gcc_matrices_full[:, :, :self.target_coils]
        compressed_hybrid_unaligned = self._apply_compression(hybrid_space_for_einsum, gcc_matrices_unaligned)

        # 4-2. Aligned 압축 적용
        gcc_matrices_aligned = self._align_gcc_matrices(gcc_matrices_full)
        compressed_hybrid_aligned = self._apply_compression(hybrid_space_for_einsum, gcc_matrices_aligned)

        # 5. 압축된 하이브리드 공간 데이터를 다시 k-space로 변환합니다.
        # 5-1. Unaligned 결과 변환
        # x 방향으로 FFT 수행 (데이터 형태가 (PE,RO,VCoils)이므로 axis=1)
        kspace_compressed_unaligned_shifted = fft.fft(compressed_hybrid_unaligned, axis=1)
        # k-space의 중심이 중앙에 오도록 fftshift 적용
        kspace_gcc = fft.fftshift(kspace_compressed_unaligned_shifted, axes=1)

        # 5-2. Aligned 결과 변환
        kspace_compressed_aligned_shifted = fft.fft(compressed_hybrid_aligned, axis=1)
        kspace_aligned_gcc = fft.fftshift(kspace_compressed_aligned_shifted, axes=1)
        
        # --- [핵심 수정 부분 끝] ---

        return kspace_gcc, kspace_aligned_gcc
        
    def compress_file(self, input_path, output_path):
        """
        단일 H5 파일을 읽어 압축을 수행하고 결과를 새 H5 파일로 저장합니다.

        Args:
            input_path (str): 원본 H5 파일 경로.
            output_path (str): 압축된 결과를 저장할 H5 파일 경로.
        """
        with h5py.File(input_path, 'r') as f_in:
            if self.kspace_dataset_name not in f_in:
                print(f"경고: '{input_path}' 파일에 '{self.kspace_dataset_name}' 데이터셋이 없어 건너뜁니다.")
                return

            original_kspace_4d = f_in[self.kspace_dataset_name][()]
            num_slices = original_kspace_4d.shape[0]

            compressed_gcc_list = []
            compressed_aligned_gcc_list = []

            for i in tqdm(range(num_slices), desc="  Processing Slices", leave=False, ascii=True):
                kspace_slice_raw = original_kspace_4d[i, :, :, :]
                kspace_slice_transposed = kspace_slice_raw.transpose(1, 2, 0) # (Coils, PE, RO) -> (PE, RO, Coils)
                
                kspace_gcc, kspace_aligned_gcc = self.process_slice(kspace_slice_transposed)
                
                # 결과 저장을 위해 차원 순서 변경: (Virtual_Coils, PE, RO)
                compressed_gcc_list.append(kspace_gcc.transpose(2, 0, 1))
                compressed_aligned_gcc_list.append(kspace_aligned_gcc.transpose(2, 0, 1))

            final_gcc_4d = np.stack(compressed_gcc_list, axis=0)
            final_aligned_gcc_4d = np.stack(compressed_aligned_gcc_list, axis=0)
            
            with h5py.File(output_path, 'w') as f_out:
                f_out.create_dataset('kspace_gcc_unaligned', data=final_gcc_4d)
                f_out.create_dataset('kspace_aligned_gcc', data=final_aligned_gcc_4d)
                if 'mask' in f_in:
                    f_out.create_dataset('mask', data=f_in['mask'][()])
                
                f_out.attrs['original_shape'] = original_kspace_4d.shape
                f_out.attrs['compressed_shape'] = final_aligned_gcc_4d.shape
                f_out.attrs['target_coils'] = self.target_coils

    def run(self, input_dir, output_dir):
        """
        입력 디렉토리의 모든 H5 파일에 대해 압축을 실행하고 출력 디렉토리에 저장합니다.

        Args:
            input_dir (str): 원본 H5 파일들이 있는 디렉토리 경로.
            output_dir (str): 압축된 파일들을 저장할 디렉토리 경로.
        """
        os.makedirs(output_dir, exist_ok=True)
        file_list = sorted([f for f in os.listdir(input_dir) if f.endswith('.h5')])
        
        if not file_list:
            print(f"'{input_dir}' 디렉토리에서 H5 파일을 찾을 수 없습니다.")
            return

        print(f"총 {len(file_list)}개의 파일에 대해 압축을 시작합니다...")

        for filename in tqdm(file_list, desc="Total Progress", ascii=True):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace('.h5', '_gcc_hybrid.h5'))
            self.compress_file(input_path, output_path)
    
        print("\n모든 파일 처리가 완료되었습니다.")
        print(f"결과는 '{output_dir}' 디렉토리에 저장되었습니다.")



def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    return torch.from_numpy(data)

class DataTransform:
    def __init__(self, isforward, max_key):
        self.isforward = isforward
        self.max_key = max_key
    def __call__(self, mask, input, target, attrs, fname, slice):
        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1
        
        kspace = to_tensor(input * mask)
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        mask = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
        return mask, kspace, target, maximum, fname, slice
