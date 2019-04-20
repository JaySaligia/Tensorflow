from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
ckpt_path= r"C:\linuxfile\tf\commodity\commodity_model.ckpt"
print_tensors_in_checkpoint_file(file_name=ckpt_path, tensor_name='', all_tensors=True, all_tensor_names=True)
