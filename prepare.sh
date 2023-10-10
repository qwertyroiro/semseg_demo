mkdir ckp
wget -P ./ckp https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
wget -P ./ckp https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
wget -P ./ckp https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
poetry run python -m spacy download en_core_web_sm