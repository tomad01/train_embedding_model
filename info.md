ssh -i ~/.ssh/cudo root@194.68.244.13
rsync -avz -e "ssh -i ~/.ssh/cudo" ./dataset.zip root@194.68.244.13:/root/
zip -r model.zip model/
rsync -avz -e "ssh -i ~/.ssh/cudo" root@194.68.244.13:/root/train_embedding_model/model.zip ./

conda env export --no-builds | grep -v "prefix" > environment.yml 
conda env create --name pyml --file=environment.yml


wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
rm -rf Miniconda3-latest-Linux-x86_64.sh

conda init bash

git config --global user.email "tomadragos96@gmail.com"
git config --global user.name "Dragos Toma"
git clone https://github.com/tomad01/train_embedding_model.git



(echo -n '{"prompt":"Describe this screenshot. Be precise, describe all elements.", "model": "phi3v", "b64_image": "'; base64 < test_dataset/51f60c02813c146e9a2c641f979bc543576d1810828e0f3f4e5dd98f0db05a24.jpg; echo '"}') | curl -H "Content-Type: application/json" -d @- --header 'Authorization: Bearer czerwona220' "https://puna.genai-stg1.hz.de.9ol.win/api/vision"