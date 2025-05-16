# Generating test vectors for validation

1. Cityscapes dataset is available at https://www.cityscapes-dataset.com/downloads/. You will need the gtFine and leftImg8bit sets.

2. Execute 
```SHELL
# go to FINN directory
cd ../finn
# get in docker container to ensure you have the correct python environment
./run_docker.sh
cd ../scripts
# generate test vectors with random input
python generate_test_vectors.py -m unet -i 1 3 256 256 -o path_to_output_directory -c ../models/state_dict.pth
# generate test vectors with random input from cityscapes
python generate_test_vectors.py -m unet -i 1 3 256 256 -o path_to_output_directory -c ../models/state_dict.pth 
--dataset cityscapes --path path_to_dataset_root
```