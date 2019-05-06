# image_colorization
SAGAN implementation in pytorch for image colorization task. 


In this work, we tackle the task of fully automated image colorization using state of the art methods in the field of deep learning for vision. Over recent years, a significant amount of work has gone into the process of automated image colorization. In our approach, we aim to use a Self Attention Generative Adversarial Network (SAGAN). The self-attention mechanism in convolutions enables the generator to maintain consistency in detailed features in different areas of the image. We demonstrate the use of perceptual loss in the generator for generating more vibrant images when compared to the more widely used L1 loss that produces overly saturated colors. We trained the model on the publicly available dataset CIFAR-10 and a self-extracted dataset from Unsplash.
