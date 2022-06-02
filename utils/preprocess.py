import numpy as np
import pickle


from util.datasets.mouse_v1.preprocess import transform_to_svd_components




def preprocess(raw_data, svd, mean):
    preprocessed_data = []
    annotation = []
    sub_seq_length = 21
    sliding_window = 5
    for i, (name, sequence) in enumerate(raw_data['annotator-id_0'].items()):
        # Preprocess sequences
        vec_seq = sequence['keypoints']
        annotation = sequence['annotations']

        # vec_seq = normalize(vec_seq)
        vec_seq = np.transpose(vec_seq, (0, 1, 3, 2))
        vec_seq, _, _ = transform_to_svd_components(
            vec_seq,
            svd_computer=svd,
            mean=mean
        )

        # Pads the beginning and end of the sequence with duplicate frames
        vec_seq = vec_seq.reshape(vec_seq.shape[0], -1)
        pad_vec = np.pad(vec_seq, ((sub_seq_length//2,
                                    sub_seq_length-1-sub_seq_length//2), (0, 0)), mode='edge')

        # Converts sequence into [number of sub-sequences, frames in sub-sequence, x/y alternating keypoints]
        sub_seqs = np.stack([pad_vec[i:len(pad_vec)+i-sub_seq_length+1:sliding_window]
                             for i in range(sub_seq_length)], axis=1)
        preprocessed_data.append(sub_seqs)
    preprocessed_data = np.concatenate(preprocessed_data, axis=0)
    return preprocessed_data
