# RAP: Risk-Aware Prompting for LLM Driving Agents: TTC-Triggered Safety Override with Trajectory Forecasting


##  Getting Started
### 1. Requirements 

```bash
conda create -n rap python=3.8 
conda activate rap
pip install -r requirements.txt
```

**Note:** rap requires specific versions of certain libraries **(i.e. `langchain==0.0.335`, `openai==0.28.1`, `chromadb==0.3.29`)**, Please adhere to the versions specified in  `requirements.txt`.

### 2. Configuration  

All configurable parameters are located in `config.yaml`.

Before running rap, set up your OpenAI API keys. rap supports both OpenAI and Azure Openai APIs. 

We used the api from https://gpt.zhizengzeng.com/.

Configure as below in `config.yaml`:
```yaml
OPENAI_API_TYPE: # 'openai' or 'azure'
# below are for Openai
OPENAI_KEY: # 'sk-xxxxxx' 
OPENAI_CHAT_MODEL: 'gpt-4-1106-preview'
# below are for Azure OAI service
AZURE_API_BASE: # https://xxxxxxx.openai.azure.com/
AZURE_API_VERSION: ""
AZURE_API_KEY: #'xxxxxxx'
AZURE_CHAT_DEPLOY_NAME: # chat model deployment name
AZURE_EMBED_DEPLOY_NAME: # text embed model deployment name  
```

### 3. Training trajectory predictor

To train the trajectory predictor, first, collect the trajectory data by running:
```bash
python run_rap.py --mode collect --traj_data_out <output_path>
```
This will generate trajectory data that can later be used to train the model.

Once the data is collected, you can train the model using:
```bash
python run_rap.py --mode train --traj_data_out <output_path> --traj_ckpt <checkpoint_path>
```

### 4. Running rap 

Running rap is straightforward:
```bash
python run_rap.py --mode run
```
The default setting runs with different seeds. You can modify this in `config.yaml`.

After completing the simulations, check the `output` folder. `log.txt` contains detailed steps and seeds for each simulation, and all simulation videos are saved here too.


#### Use reflection module:

To activate the reflection module, set `reflection_module` to True in `config.yaml`. New memory items will be saved to the updated memory module.


## Reference

RAP is built upon the DiLu project(https://github.com/PJLab-ADG/DiLu) for LLM-based autonomous driving. The original DiLu repository was adapted to include a trajectory prediction model and a TTC-triggered safety override mechanism.

Citation for DiLu:
```bibtex
@article{wen2023dilu,
  title={Dilu: A knowledge-driven approach to autonomous driving with large language models},
  author={Wen, Licheng and Fu, Daocheng and Li, Xin and Cai, Xinyu and Ma, Tao and Cai, Pinlong and Dou, Min and Shi, Botian and He, Liang and Qiao, Yu},
  journal={arXiv preprint arXiv:2309.16292},
  year={2023}
}
```

## 📝 License
rap is released under the Apache 2.0 license.
