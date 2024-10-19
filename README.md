# VERBAL TO VISIUAL - TEXT TO IMAGE
In this project we have explored different models for text - to -images generation in which we take in text it is is converted into its embedding and then we get our image generated

## USING XL+GANS
INPUT = "Yellow flower with black stemen"

OUTPUT:
![alt text](https://github.com/Captain-MUDIT/Verbal-to-Visual/blob/main/Screenshot%202024-10-19%20043938.png)

INPUT="red flower"

OUTPUT:
![alt text](image-1.png)

INPUT = "a pink flower"

OUTPUT:
![alt text](image-2.png)

USING STACKGANS/Stage1:
OUTPUT:

flower dataset

![alt text](image-3.png)
![alt text](image-4.png)
![alt text](image-5.png)
![alt text](image-6.png)

flickr8k:
![alt text](image-7.png)

## APPROACHES AND THEORY
So in our project we explored the approches of gans like stack gans , dc gans , XLnets+gans , and we also explored modern hopefield 
model sp lets see all the approches

### DCGANS
* In our project, DCGAN serves as the core architecture for both the generator and the discriminator.
* The generator uses convolutional layers to upsample a noise vector, conditioned by the text embeddings. This allows the generator to create structured images that correspond to the text input, like generating a "blue sky with clouds."
* The discriminator is a CNN-based model that learns to distinguish between real images (from the dataset) and the generated images. It helps the generator improve by providing feedback on how realistic the images are.
* Contribution: DCGAN’s convolutional layers allow the model to generate images with local coherence, capturing detailed patterns and structures. The use of DCGAN in your project improves the generation of high-quality images by leveraging deep convolutional networks, which are more effective at generating visually consistent and realistic images.

### STACK GANS
![alt text](image-8.png)
* Stage1 : The model converts this description into an embedding vector using a pre-trained text encoder (like a pre-trained language model). This embedding is then passed to the first stage of StackGAN, which generates a low-resolution image (e.g., 64x64 pixels) that roughly captures the overall shape, color, and basic structure of the object in the description.  
* Stage2 :  This low-resolution image is passed into the second stage of StackGAN, along with the text embedding again. Here, the model refines the image, adding finer details like textures and lighting, and increases the resolution (e.g., 256x256 pixels), here residual blocks are used to capture the features more . This results in a more photorealistic and detailed image that matches the textual description.
* Contribution: StackGAN enables your project to generate progressively refined images, with each stage improving the quality and resolution, making the generated images more realistic and aligned with the input text.
### MORDERN HOPFIELD NETWORK
![alt text](image-9.png)
* Modern Hopfield networks are used to store and retrieve complex patterns (in this case, embeddings) that correspond to text-image relationships. These networks iteratively update the embeddings to retrieve the most relevant stored patterns, which helps improve the fidelity of the generated images.
* In our project, after obtaining an initial embedding from the text description (e.g., using XLNet), a Hopfield network layer might be used to refine or retrieve related patterns that can aid in the generation of images with more coherent details. The modern Hopfield network excels at associating and recalling these complex relationships, such as specific texture patterns, object shapes, or even lighting conditions described in the text.
* Contribution: The use of modern Hopfield networks in your project ensures that the model can effectively recall detailed image patterns related to the input description. This results in more consistent and accurate generation, as the model can better preserve high-dimensional relationships between text and images.

[##Reasearch papper for mhn](https://arxiv.org/pdf/2208.04441)

### XLNets + gans
* Making the use of XLNet and GANs, we were able to generate the images of flowers based on prompts. 
Where XLNet was used to generate the text embeddings from the captions present in the dataset.
* GANs generated the image based on the text embeddings they were fed.
* The combination of GANs + XLNet proves to be effective    because the system can simultaneously learn from textual and visual data, enhancing its ability to generate images that are contextually aligned with the input text. 
* The discriminator in GANs provides feedback to the generator, allowing it to refine the generated images iteratively. When the generator uses high-quality text embeddings from XLNet as input, this feedback loop becomes more effective, leading to better convergence and higher-quality outputs
XLNet outperformed BERT, because even using the BERT model we were not able to get an image which corresponds to the prompt.

## DATASETS
* [flickr30k](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset)
* [hugging face flower deataset](https://huggingface.co/datasets/pranked03/flowers-blip-captions/viewer/default/train?p=62)
  * way to download :
    ```python
    from datasets import load_dataset
    # Load the Flowers BLIP Captions dataset
    dataset = load_dataset("pranked03/flowers-blip-captions")
     dataset.save_to_disk("flowers-blip-captions")

* [oxford 102 flower dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)

### FUTURE SCOPE
* The stage 2 of staqck gans due to memory issue and limitted availibility of gpu access is still pending so our aim is to that in near future 
* MHN model due to less availibilty of resources is still remaining so we plan to do more resources in that also
* Training on more dataset like the coco bird dataset will be our next step.
* Reducing noise of the the output by using svre 

### MENTORS
[Rohan Parab](https://github.com/Rohan20-10)

[Param Parekh](https://github.com/Param1304)

### CONTRIBUTERS

[Janvi Soni]()

[Yadnyesh Patil](https://github.com/YoLynx)

[Mudit Jain]()




