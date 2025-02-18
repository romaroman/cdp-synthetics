1. [Download the dataset](https://github.com/romaroman/cdp-synthetics-dataset)
2. Set ```CDPSYNT_PROJECT_DIR``` env var to **this directory path**
3. Set ```CDPSYNT_DATA_DIR``` env var to **downloaded dataset path**
4. Install requirements with ```pip install -r requirements.txt```
5. Configure ```config/general.yaml``` to your needs or create another config
   1. Specifically you must change io.dir_root to point to the path of the dataset. 
   2. You can also pass the configuration params to the script ```python scripts/main.py ... dataset.device=iPXS```
6. Run ```python scripts/main.py --config-name general``` or with other ```--config-name```
7. The checkpoints  will be saved in the ```outputs``` directory inside ```CDPSYNT_DATA_DIR``` directory
8. After training is done run the same script with ```--mode test``` to test the model
9. The testing results will be saved in the ```outputs``` directory inside ```CDPSYNT_DATA_DIR``` as CSV