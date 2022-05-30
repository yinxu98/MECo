# MECo
Code for "Multiple Embeddings Contrastive Pre-Training for Remote Sensing Image Classification"

### Abstract
> This letter focuses on remote sensing image interpretation and aims to promote the use of contrastive selfsupervised learning in varied applications of remote sensing image classification. The proposed method is a contrastive selfsupervised pre-training framework that encourages the network to learn image representations by comparing image embeddings extracted by different encoders and predictors. Experiments were carried out on a variety of remote sensing image datasets to determine the efficacy of the proposed method for classification tasks. Results show that the proposed framework exploits the capabilities of encoders and outperforms the supervised learning method in terms of classification accuracy. Besides, it takes a few pre-training epochs to find a suboptimal initialization of network weights, and the pre-trained encoders use a little training data to get outstanding classification results, which shows the time and data efficiency of the proposed framework.

### Code
Our code is based on the core of MMDetection. We use config files to manage the experimental settings ([MECo/cls/configs](MECo/cls/configs)).

[MECo/work_dirs](MECo/work_dirs) stores the pretrained models.

### Pre-Train
To pre-train an encoder with the proposed **MultiEmbedding** framework on the OpenSARShip dataset using GPU 0, run the command under [MECo/cls](MECo/cls) folder:

```
./scripts/pretrain_single.sh multiembedding opensarship 0
```

### Evaluate
To evaluate an pre-trained encoder in fine-tuning mode on the RSSCN7 dataset using GPU 1, run the command under [MECo/cls](MECo/cls) folder:

```
./scripts/test.sh rsscn single ft 1
```

To evaluate an pre-trained encoder in linear probing mode on the RSSCN7 dataset using GPU 2, run the command under [MECo/cls](MECo/cls) folder:

```
./scripts/test.sh rsscn single lin 2
```
