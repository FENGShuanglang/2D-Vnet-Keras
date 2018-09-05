### *Keras implementation of 2D Vnet*
this is a 2D Vnet model of Keras Version,The architecture of the model is the same as the original 3D Vnet, except that this is a network for 2D image segmentation (the convolution kernel becomes 2D)

### *The visual model is as follows(The convolutional layer in the figure should be two-dimensional)*
![Model](https://github.com/FENGShuanglang/2D-Vnet-Keras/blob/master/VNetDiagram.png)

### *Parameter Settings*
* ```stage_num```:allows you to change the stage of the network. For example, the network model in the picture is 5 stage. You can set the ```stage_num=6``` to make the network deeper into a stage. The default parameter is 5
