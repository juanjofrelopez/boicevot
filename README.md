# BoiceVot

## How to

- use python 3.9

```
pyenv local 3.9
```

- run install.sh

```bash
./install.sh
```

- source the venv

```bash
source .venv/bin/activate
```

- launch the pgvector and llamafile docker instances

```bash
cd infra/
sudo docker compose up
```

- create the embeddings

```bash
cd ..
python create_embeddings.py
```

- run the main script

```bash
python main.py
```
