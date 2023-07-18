# Deploying model on Alveo U250

The driver uses the pynq library and XRT to deploy and access the design on the board. 

To evaluate execution performances on empty inputs, run `driver/throughput_test.sh`.

To validate the design, first generate test vectors using `/scripts/generate_test_vectors.py` and provided model files and weights. Then execute:
```SHELL
python validate_segsem.py --test_input path_to_input.npy --test_output path_to_expected_output.npy
```
