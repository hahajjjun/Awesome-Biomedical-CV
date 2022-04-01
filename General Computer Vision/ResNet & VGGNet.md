# Computer Vision Week 1. [2021 CV]

### ResNet : Deep Residual Learning for Image Recognition

- **Abstract**
    - Deeper neural networks → vanishing/exploding gradient, degradation, overfitting... problematic, difficult to train!
    - Residual learning framework consists of residual block → ease the **training** of deeper networks
    - Explicit reformulation of layers : learning residual functions(reference to the layer inputs) instead of unreferenced function
    - Residual learning → Easier to optimize, gain accuracy from considerably increased depth
    - ImageNet dataset : 152 layers residual nets evaluated(ensemble of residual nets achieves 3.57% error) ← 8X deeper than VGG nets but lower complexity
    - ILSVRC 2015 classification task 1st / CIFAR-10 with 100, 1000 layers
    - Due to extremely deep representations → COCO object detection dataset 28% relative improvement
    - ILSVRC & COCO 2015 competitions / ImageNet detection / ImageNet localization / COCO detection / COCO segmentation
- **Intro**
    - DCNN → breakthroughs for image classification, different levels of features(low/mid/high) can be enriched by depth
    - **ICLR 2015 VGG[16], CVPR 2015 GoogleNet[22], BatchNorm[30], PReLu-Net & He initialization** : benefited from very deep models
        
        <img src = "https://user-images.githubusercontent.com/75057952/151806173-8fb7dbf8-dfa6-4bcb-ace7-470bd584a77f.png" width = "500dp"/>
        
    
    <aside>
    ❓ Layer를 깊게 쌓으면 쌓을수록 좋은 Network를 학습할 수 있을까?
    
    </aside>
    
    - Problems
        - Vanishing / Exploding Gradients problem → Usually addressed by **normalized initialization & intermediate normalization layers(BatchNorm)**
            
            [Weight Initialization in a Deep Network (C2W1L11)](https://www.youtube.com/watch?v=s2coXdufOzE)
            
            - Deep neural network에서 Layer 개수가 매우 커지게 되면 activation value가 exponentially increase or decrease
                - 마찬가지로 its gradient 역시 exponentially increase or decrease
            - Partial solution of vanishing / exploding gradients problem : weight initialization in a deep neural network
                - Weight Initialization의 기본 원리는, 이전 layer의 node 개수가 많아질수록 $z = \sum w_ix_i$의 값이 커지고 $\sigma(z)$를 계산할 때 z가 큰 range에서 sigmoid를 activation으로 사용할 때 gradient가 굉장히 작아졌었다. ReLU function도 마찬가지로 여러 Layer 거치면서 작은 값들이 곱해지면 vanishing gradient 문제에 빠질 수 있다.
                - $Var(W) \propto \sqrt{1 \over n_{in}}$
                    - LeCun initialization
                    - He initialization
                    - Xavier initialization
        - Deeper networks → Degradation problem exposed
            - Layer 깊이가 깊어짐에 따라 training, test error가 감소하지 않고 증가
                
                
                <img src = "https://user-images.githubusercontent.com/75057952/151806184-bb780e17-3d4c-457d-a4c7-d4b38a4928f6.png" width = "500dp"/>
                
            - Degradation is **not** caused by **overfitting**
            - Adding more layers to a suitably deep model leads to **higher training error**
            - Degradation problem indicates that not all systems are similarly easy to optimize
    - Idea
        - Shallow model + add more layers → deeper counterpart
            - If the added layers are identity mapping, and other layers are copied from the learned shallower model
                
                ⇒ Constructed solution indicates that a deeper model should produce no higher training error than its shallower counterpart
                
        - Degradation problem 해결을 위해 **Deep residual learning framework** 제안
        - desired mapping H(x)를 학습하는 대신, F(x) = H(x) - x를 학습함, original mapping H(x) = F(x) + x
        
        <aside>
        ⚠️ Hypothesis : It is easier to optimize the residual mapping than to optimize the original, unreferenced mapping
        
        </aside>
        
        - Extreme case : 만약 identity mapping is optimal → residual = 0
    - Residual Block
        - Formulation of F(x) + x : by feedforward nerual networks with **shortcut connections**
        - Architecture of Residual Block
            
            <img src = "https://user-images.githubusercontent.com/75057952/151806189-143bf104-434f-4166-a244-fa2a37ae9fa3.png" width = "400dp"/>
            
            - 잔차 함수 F(x)를 학습하는 layer + layer를 건너뛰어 input 값을 더해 주는 shortcut connection으로 구성됨
            - F(x)와 x를 더한 결과로 도출되는 H(x)가 다음 residual block의 input이 됨
            - $y = F(x) + x = W_2\sigma(W_1x) + x$
            - shortcut connection으로는 identity mapping을 사용해도 무방, 추가적인 parameter 증가가 없고 구현이 간단하다는 장점
    - ImageNet experiments : Show that...
        - **Extremely deep residual nets are easy to optimize, but the counterpart plain nets exhibit higher training error when depth increases**
        - **Deep residual nets can easily benefit from greatly increased depth, producing substantially better results that previous networks**
        - 152-layer residual net was the deepest while lower complexity than VGG nets
        - Ensemble has 3.57% top-5 error
    - CIFAR-10 dataset에서도 >100, >1000 layer model train하면서 similar phenomena 관찰
    - ImageNet detection, ImageNet localization, COCO detection, COCO segmentation
- **Related Work**
    - Residual Representation
    - **Shortcut connection**
        - **Connection skipping one or more layers**
        - **Few intermediate layers directly connected to auxiliary classifiers for addressing vanishing/exploding gradients.(GoogLeNet)**
            
            <img src = "https://user-images.githubusercontent.com/75057952/151806189-143bf104-434f-4166-a244-fa2a37ae9fa3.png" width = "400dp"/>
            
    - **Highway networks**
        - **ResNet과 마찬가지로 깊은 네트워크 학습에 용이**
        - **T(Transform), C(Carry) Gating function을 통해 shortcut connection과 유사한 기능 수행**
        - **이것과 비교하면 Shortcut connection은 always-open gate : C=1**
            
            <img src = "https://user-images.githubusercontent.com/75057952/151806189-143bf104-434f-4166-a244-fa2a37ae9fa3.png" width = "400dp"/>
            
            <img src = "https://user-images.githubusercontent.com/75057952/151806306-06f63437-9df9-4a7c-b205-d35526f1154e.png" width = "400dp"/>
            
- **Deep Residual Learning Architecture**
    - Residual Learning & Identity mapping by shortcuts
        - Residual Learning : Explicitly define $F(x) = H(x) - x$ as residual function
            - Expect stacked layers to approximate F(x)
            - This reformulation is motivated by degradation problem
                - Intuitively, **deeper model constructed by adding layers with identity mappings** should have training error not greater than its **shallower counterpart.**
            - Approximating identity mappings by multiple nonlinear layers → might have difficulties!
            - However, as optimal function is closer to an identity mapping than a zero mapping, it should be easier to find the perturbations with reference to an identity mapping than to learn the function as a new one
        - Shortcut connection ⇒ identity mapping vs projection shorcuts
            
            <img src = "https://user-images.githubusercontent.com/75057952/151806189-143bf104-434f-4166-a244-fa2a37ae9fa3.png" width = "400dp"/>
            
            - Generally, $y = F(x) + x = W_2\sigma(W_1x) + x$     ⇒      $F(x, \{W_i\})+W_sx$ : multiple convolutional layers
            - $W_s$ : Identity mapping(1)이거나, x와 y의 dimension 다를 때 linear projection을 통해 dimesnion을 맞춤(2)
    - Network Architectures
        - Plain Network vs Residual Network
            - Plain(Mainly inspired by the philosophy of VGG) ⇒ Residual Block
            - 34-layer plain network : have 3X3 convolutional filters & follow **two design rules**
                - For the same output feature map size, the layers have the same number of filters
                - If the feature map size is halved, the number of filters is doubled : preserve time complexity/layer
                - Downsampling : max pooling with stride 2
                - Ends with a global average pooling layer and 1000-way fully-connected layer with softmax
                    
                    <img src = "https://user-images.githubusercontent.com/75057952/151806194-94e7eeef-4b1e-49d8-bf02-4fff18c8b6dc.png" width = "500dp"/>
                    
                - Plain(3.6bil FLOPs) : Residual(3.6bil FLOPs : 18% of VGG-19)
            - Residual Network
                - Insert shortcut connections to plain network, turning plain network to its counterpart residual version
                - Identity shortcuts are inserted
                - When dimension **increases : Two Options available. Which one is better?**
                    - Shortcut performs identity mapping, with extra zero entries padded for increased dimensions (No extra param)
                    - Apply 1X1 convolution(W_s) to match dimensions : $y = F(x,\{W_i\}) + W_sx$ (Projection)
                
        - Resnet.py
            
            [torchvision.models.resnet - Torchvision 0.8.1 documentation](https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html)
            
            - Setting
                
                ```python
                import torch
                import torch.nn as nn
                from .utils import load_state_dict_from_url
                
                __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
                           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
                           'wide_resnet50_2', 'wide_resnet101_2']
                
                model_urls = {
                    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
                    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
                    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
                    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
                    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
                    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
                    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
                    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
                    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
                }
                
                def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
                    """3x3 convolution with padding"""
                    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                     padding=dilation, groups=groups, bias=False, dilation=dilation)
                
                def conv1x1(in_planes, out_planes, stride=1):
                    """1x1 convolution"""
                    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
                ```
                
            - BasicBlock Class = Residual Block
                
                <img src = "https://user-images.githubusercontent.com/75057952/151806189-143bf104-434f-4166-a244-fa2a37ae9fa3.png" width = "400dp"/>
                
                ```python
                class BasicBlock(nn.Module):
                    expansion = 1
                
                    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                                 base_width=64, dilation=1, norm_layer=None):
                        super(BasicBlock, self).__init__()
                        if norm_layer is None:
                            norm_layer = nn.BatchNorm2d
                        if groups != 1 or base_width != 64:
                            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
                        if dilation > 1:
                            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
                        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
                        self.conv1 = conv3x3(inplanes, planes, stride)
                        self.bn1 = norm_layer(planes)
                        self.relu = nn.ReLU(inplace=True)
                        self.conv2 = conv3x3(planes, planes)
                        self.bn2 = norm_layer(planes)
                        self.downsample = downsample
                        self.stride = stride
                
                    def forward(self, x):
                        identity = x
                
                        out = self.conv1(x)
                        out = self.bn1(out)
                        out = self.relu(out)
                
                        out = self.conv2(out)
                        out = self.bn2(out)
                
                        if self.downsample is not None:
                            identity = self.downsample(x)
                
                        out += identity
                        out = self.relu(out)
                
                        return out
                ```
                
            - Bottleneck Class
                
                ```python
                class Bottleneck(nn.Module):
                    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
                    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
                    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
                    # This variant is also known as ResNet V1.5 and improves accuracy according to
                    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
                
                    expansion = 4
                
                    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                                 base_width=64, dilation=1, norm_layer=None):
                        super(Bottleneck, self).__init__()
                        if norm_layer is None:
                            norm_layer = nn.BatchNorm2d
                        width = int(planes * (base_width / 64.)) * groups
                        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
                        self.conv1 = conv1x1(inplanes, width)
                        self.bn1 = norm_layer(width)
                        self.conv2 = conv3x3(width, width, stride, groups, dilation)
                        self.bn2 = norm_layer(width)
                        self.conv3 = conv1x1(width, planes * self.expansion)
                        self.bn3 = norm_layer(planes * self.expansion)
                        self.relu = nn.ReLU(inplace=True)
                        self.downsample = downsample
                        self.stride = stride
                
                    def forward(self, x):
                        identity = x
                
                        out = self.conv1(x)
                        out = self.bn1(out)
                        out = self.relu(out)
                
                        out = self.conv2(out)
                        out = self.bn2(out)
                        out = self.relu(out)
                
                        out = self.conv3(out)
                        out = self.bn3(out)
                
                        if self.downsample is not None:
                            identity = self.downsample(x)
                
                        out += identity
                        out = self.relu(out)
                
                        return out
                ```
                
            - ResNet Class
                
                ```python
                class ResNet(nn.Module):
                
                    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                                 norm_layer=None):
                        super(ResNet, self).__init__()
                        if norm_layer is None:
                            norm_layer = nn.BatchNorm2d
                        self._norm_layer = norm_layer
                
                        self.inplanes = 64
                        self.dilation = 1
                        if replace_stride_with_dilation is None:
                            # each element in the tuple indicates if we should replace
                            # the 2x2 stride with a dilated convolution instead
                            replace_stride_with_dilation = [False, False, False]
                        if len(replace_stride_with_dilation) != 3:
                            raise ValueError("replace_stride_with_dilation should be None "
                                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
                        self.groups = groups
                        self.base_width = width_per_group
                        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                               bias=False)
                        self.bn1 = norm_layer(self.inplanes)
                        self.relu = nn.ReLU(inplace=True)
                        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                        self.layer1 = self._make_layer(block, 64, layers[0])
                        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                                       dilate=replace_stride_with_dilation[0])
                        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                                       dilate=replace_stride_with_dilation[1])
                        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                                       dilate=replace_stride_with_dilation[2])
                        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                        self.fc = nn.Linear(512 * block.expansion, num_classes)
                
                        for m in self.modules():
                            if isinstance(m, nn.Conv2d):
                                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                                nn.init.constant_(m.weight, 1)
                                nn.init.constant_(m.bias, 0)
                
                        # Zero-initialize the last BN in each residual branch,
                        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
                        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
                        if zero_init_residual:
                            for m in self.modules():
                                if isinstance(m, Bottleneck):
                                    nn.init.constant_(m.bn3.weight, 0)
                                elif isinstance(m, BasicBlock):
                                    nn.init.constant_(m.bn2.weight, 0)
                
                    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
                        norm_layer = self._norm_layer
                        downsample = None
                        previous_dilation = self.dilation
                        if dilate:
                            self.dilation *= stride
                            stride = 1
                        if stride != 1 or self.inplanes != planes * block.expansion:
                            downsample = nn.Sequential(
                                conv1x1(self.inplanes, planes * block.expansion, stride),
                                norm_layer(planes * block.expansion),
                            )
                
                        layers = []
                        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                            self.base_width, previous_dilation, norm_layer))
                        self.inplanes = planes * block.expansion
                        for _ in range(1, blocks):
                            layers.append(block(self.inplanes, planes, groups=self.groups,
                                                base_width=self.base_width, dilation=self.dilation,
                                                norm_layer=norm_layer))
                
                        return nn.Sequential(*layers)
                
                    def _forward_impl(self, x):
                        # See note [TorchScript super()]
                        x = self.conv1(x)
                        x = self.bn1(x)
                        x = self.relu(x)
                        x = self.maxpool(x)
                
                        x = self.layer1(x)
                        x = self.layer2(x)
                        x = self.layer3(x)
                        x = self.layer4(x)
                
                        x = self.avgpool(x)
                        x = torch.flatten(x, 1)
                        x = self.fc(x)
                
                        return x
                
                    def forward(self, x):
                        return self._forward_impl(x)
                ```
                
                ```python
                def _resnet(arch, block, layers, pretrained, progress, **kwargs):
                    model = ResNet(block, layers, **kwargs)
                    if pretrained:
                        state_dict = load_state_dict_from_url(model_urls[arch],
                                                              progress=progress)
                        model.load_state_dict(state_dict)
                    return model
                ```
                
        
        `def ResNet18():
            return ResNet(BasicBlock, [2, 2, 2, 2])
        def ResNet34():
            return ResNet(BasicBlock, [3, 4, 6, 3])
        def ResNet50():
            return ResNet(Bottleneck, [3, 4, 6, 3])
        def ResNet101():
            return ResNet(Bottleneck, [3, 4, 23, 3])
        def ResNet152():
            return ResNet(Bottleneck, [3, 8, 36, 3])`
        
        <img src = "https://user-images.githubusercontent.com/75057952/151806311-c9c7fc73-9bd0-4430-bb46-a610a9f97a10.png" width = "500dp"/>
        
    - Implementation
        - Implementation for ImageNet
            - Image resize for scale augmentation
            - 224X224 crop randomly sampled from an image or its horizontal flip
            - Standard color augmentation
            - Batch Normalization right after each convolution and before activation
            - Initialize the weights(He initialization)
            - SGD(Stochastic gradient descent) with mini-batch size of 256
            - Learning rate를 0.1에서부터 10으로 나눠가며 단계적으로 줄이는 technique
            - 6X10^5 iters
            - Weight decay = 0.0001, momentum = 0.9
            - Do no use dropout
            - Adopt standard 10-crop testing & Average the scores at multiple scales for best results
- **Experiments**
    - ImageNet Classification
        - ImageNet 2012 Classification dataset : 1000 classes
        - 1.28 mil training images + 50k validation images + 100k test images
        - top-1 and top-5 error rates evaluation
        - Plain Network vs ResNet(18-layer, 34-layer)
            
            <img src = "https://user-images.githubusercontent.com/75057952/151806311-c9c7fc73-9bd0-4430-bb46-a610a9f97a10.png" width = "500dp"/>
            
            **Plain networks : 18-layer vs 34-layer**
            
            - 34-layer plain net has higher training error throughout the whole procedure
                
                <img src = "https://user-images.githubusercontent.com/75057952/151806318-163939bf-d599-4e24-a197-0d48016a605e.png" width = "500dp"/>
                
                * Bold : Validation / Light : Train Error
                
            - Solution space of the 18-layer plain network $V_{18}$ ≤ Solution space of the 34-layer plain network $V_{34}$
            - Optimization difficulty is **unlikely to be caused by vanishing gradients : Neither forward/backward signals vanish**
                - Plain networks are trained with BN : ensures forward propagated signals to have non-zero variances
                - Also verified that backward propagated gradients exhibit healthy norms with BN
                - 34-layer plain net still achieves competitive accuracy : suggesting that solver works to some extent
                    
                    <img src = "https://user-images.githubusercontent.com/75057952/151806321-ed5b76f2-3292-488f-ae4f-17000054e578.png" width = "400dp"/>
                    
                - **Deep plain nets may have exponentially low convergence rates : hurdles in reducing train error**
            
            **Residual Networks : 18-layer and 34-layer ResNets**
            
            - Same baseline architecture, only shortcut connection is added to each pair of 3X3 filters
            - Identity mapping for all shortcuts and zero-padding for increasing dimensions(no extra params)
                
                <img src = "https://user-images.githubusercontent.com/75057952/151806325-9ce6a029-42be-442a-8b9a-e28ad5c631d9.png" width = "500dp"/>
                <img src = "https://user-images.githubusercontent.com/75057952/151806321-ed5b76f2-3292-488f-ae4f-17000054e578.png" width = "500dp"/>
                
                
                
            - 34 layers ResNet exhibits considerably lower training error and is generalizable to the validation data
            - Degradation problem is well addressed in this setting, able to obtain accuracy gain from increased depth
            - Reduced top-1 error by 3.5%(34 layers) & successfully reduced training error
            - 18-layer plain / ResNet were comparably accurate, while 18-layer ResNet converges faster at early stage
                - ResNet benefits not overally deep(18 layers here) by easing the optimization by providing faster convergence at early stage
                    
                    **Q) Why early stage convergence is a benefit? 그림에서 보면 30e^4 iteration즈음으로 전반적인 convergence는 유사하지 않나요?**
                    
                    <img src = "https://user-images.githubusercontent.com/75057952/151806330-6ac5f798-808e-486d-8d49-3ba138de2fd2.png" width = "500dp"/>
                    
        - Identity vs Projection Shortcuts
            - **Concept of Identity & Projection Shortcuts**
                
                <img src = "https://user-images.githubusercontent.com/75057952/151806189-143bf104-434f-4166-a244-fa2a37ae9fa3.png" width = "400dp"/>
                
                - **input x와 F(x)를 비교했을 때 F(x)의 dimension이 x의 dimension보다 클 경우에는 두 가지 옵션이 있음**
                    - **x를 F(x)가 속해 있는 vector space에 projection해서 dimension을 늘려 주거나 ⇒ Projection Shortcut**
                    - **x에 zero-padding을 통해서 dimension을 늘린 후에 단순히 identity mapping을 적용 ⇒ Zero-padding Shortcut(params conserved)**
            - 바로 위의 18-layer, 34-layer ResNet experiment에서는 identity shortcut을 사용하여 training 성능을 향상시킬 수 있음을 관찰함
            - Identity Shortcut과 Projection Shortcut method를 비교 실험
                - (A) : dimension이 증가할 때에만 zero-padding shortcuts, 그 외에는 identity shortcut ⇒ All shortcuts are parameter-free
                - (B) : dimension이 증가할 때에만 projection shortcuts 사용하고, 그 외에는 identity shortcut
                - (C) : all shortcuts are projections
                    
                    <img src = "https://user-images.githubusercontent.com/75057952/151806335-1e7b83ca-9985-40fe-8b55-b5847908ee9f.png" width = "400dp"/>
                    
                - slightly A < B < C
                - B is slightly better than A : zero-padded dimensions in A have no residual learning
                - C is marginally better than B : extra parameters introduced by projections shortcuts 덕분이라고 설명함
                - 그러나 이러한 차이가 미미하고, 앞선 실험에서도 나왔듯이 zero-padding shortcut만을 사용해도 degradation problem 해결할 수 있었음 ⇒ paper에서는 memory/time complexity & model size 측면에서 효율적인 A,B 옵션만을 사용함
                - **특히, Identity shortcut(zero-padding)은 bottleneck architecture에서 complexity를 증가시키지 않음**
        - Deeper Bottleneck Architectures
            - Deeper nets에서 Residual Block의 Architecture를 bottleneck 구조로 설계
            - 18,34-layer ⇒ 50,101,152-layer(Bottleneck Architecture)
            - Modification of building block (concerns on the **training time**)
                
                <img src = "https://user-images.githubusercontent.com/75057952/151806338-338527fe-c9c5-49b6-b53b-c7bfc42b720e.png" width = "400dp"/>
                
                - 3X3 layer 앞뒤의 1X1 layer가 dimension을 reduce했다가 다시 늘리는 역할을 하기 때문에, 가운데 3X3 layer는 더 작은 input/output dimension을 가지게 됨 : dimension bottleneck
            - Bottleneck architecture에서 parameter-free identity shortcut의 impact가 더욱 분명함
                - Bottleneck architecture의 identity shortcut을 projection으로 바꾸면
                    
                    ⇒ projection shortcut이 high-dimensional end를 서로 연결하고 있으므로 time complexity & model size 가 증가하는 효과가 배가됨
                    
                    . If the identity shortcut in Fig. 5 (right) is replaced with projection, one can show that the time complexity and model size are doubled, as the shortcut is connected to the two high-dimensional ends.
                    
                - Identity shortcut은 Bottleneck architecture에서 더욱 효과적
            - 50-layer ResNet architecture : replace 2-layer block into 2-layer bottleneck block, 3.8 billion FLOPs(34-layer ResNet과 큰 차이 없음)
            - 101-layer ResNet architecture : 7.6 bil FLOPs / 152-layer ResNet architecture : 11.3 bil FLOPs < VGG-16/19 nets : 15.3/19.6 bil FLOPs
                
                <img src = "https://user-images.githubusercontent.com/75057952/151806311-c9c7fc73-9bd0-4430-bb46-a610a9f97a10.png" width = "500dp"/>
                
                <img src = "https://user-images.githubusercontent.com/75057952/151806335-1e7b83ca-9985-40fe-8b55-b5847908ee9f.png" width = "500dp"/>
                
                [GitHub - KaimingHe/resnet-1k-layers: Deep Residual Networks with 1K Layers](https://github.com/KaimingHe/resnet-1k-layers)
                
        - Comparisons with SOTA models
            
            <img src = "https://user-images.githubusercontent.com/75057952/151806344-b3d91566-6402-4f41-8ab1-fb29eca29297.png" width = "500dp"/>
            
    - CIFAR-10
        - 50k training images, 10k testing images in 10 classes
        - 32X32 image inputs
        - 논문의 저자들은 CIFAR-10 dataset을 위한 simple architecture를 따로 만들어서 experiment를 수행했음
            - SOTA model과의 직접적인 성능비교가 아닌, 성능 대비 parameter 개수 면에서 효율적 + Deep network임에도 training이 쉽다는 점 강조
            - CIFAR-10 ResNet에 관한 github 보면 ResNet50 architecture를 그대로 사용하되 input size만 바꿔서 활용하고 있음
            - [https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py](https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py)
        - Plain architecture(ResNet20 Counterparts) → Identity shortcuts in all cases → ResNet20
            - 32X32 image input ⇒ first 3X3 convolution layer ⇒ **2n** **convolution layers with** 32X32 output map&16 channels ⇒ **2n convolution layers with** 16X16 output map&32 channels ⇒ **2n convolution layers with** 8X8 output map&64 channels ⇒ 10-way fully-connected layer&softmax (total **6n+2)**
            - **3n** identity shortcuts for ResNet20
                
                <img src = "https://user-images.githubusercontent.com/75057952/151806360-f42e8961-9515-454d-9dee-92cb09bd54ac.png" width = "500dp"/>
                
        - Implementations
            - weight decay = 0.0001, momentum = 0.9
            - He initialization, Batch Normalization but with no dropout
            - start with learning rate of 0.1, divide it by 10 at 32k, 48k iterations and terminate training at 64k iterations
            - 50k image ⇒ 45k + 5k train/val split
            - 4 pixels padded & 32X32 random crop from padded image or its horizontal flip
        - Experiments
            - compare n= {3,5,7,9} for 6n+2 layer networks {20,32,44,56}
            - Training on CIFAR-10
                
                <img src = "https://user-images.githubusercontent.com/75057952/151806348-306f8972-ef3d-4baf-85d1-27574d60d61b.png" width = "600dp"/>
                
                - Deep plain nets suffer from degradation problem : increased depth ⇒ higher training error (ImageNet, MNIST 데이터셋에서도 유사)
                - Behavior of ResNets : Manage to overcome the optimization difficulty & accuracy gains when the depth increases
                    
                     <img src = "https://user-images.githubusercontent.com/75057952/151806352-2f8b2791-1aac-47c5-9fff-4b6357637773.png" width = "400dp"/>
                    
                - SOTA model과 비교해 보면 성능대비 parameter 수가 적음
                - n=18인 110-layer model의 경우에는 initial learning rate 0.1이 converging하기에 너무 커서 several epoch 후에야 converge 시작함
                    - 그래서 training error<80%(400 iter)될 때까지 learning rat = 0.01로 줄였다가 다시 0.1로 높여서 training 진행
                    - 110-layer network converges well but fewer parameters than SOTA models, such as FitNet & Highway
            - Analysis of Layer Responses
                - Standard deviations of layer responses
                - response = output of each 3X3 layer, after **BN** and before **ReLU/addition**
                - ResNets have generally smaller responses than their plain counterparts
                    - Supports that the residual functions might be generally closer to zero than the non-residual functions
                    - Individual layer of ResNets tend to modify the signal less
            - Exploring over 1000 layers
                - n=202, 1202-layer network
                - no optimization difficulty
                - achieve training error < 0.1%
                - 1202-layer network and 110-layer network have similar training error but testing result is worse in 1202-layer network : overfitting
                - 1202-layer network : unnecessarily large(19.4M) for small CIFAR dataset
                - maxout, dropout을 비롯한 strong regularization이 CIFAR dataset에 적용될 때 성능 향상 보고되어 있지만, ResNet에서는 오직 architecture만으로 optimization 성능을 향상시킴 ⇒ combining with stronger regularization will be studied in the future
    - Object Detection on PASCAL and MS COCO
        - ResNet has good generalization performance on other recognition tasks
        - PASCAL VOC 2007/2012 & COCO
        - VGG-16을 backbone network로 하는 Faster R-CNN과 ResNet-101의 비교
            - Average Precision 향상
            
            <img src = "https://user-images.githubusercontent.com/75057952/151806354-790141a3-f255-42b7-bcb8-d526fb4e0c8e.png" width = "400dp"/>
            

### VGG : Very Deep Convolutional Networks For Large-Scale Image Recognition

- 7X7 convolutional filter ⇒ three 3X3 convolutional filter stack
    - receptive field는 같지만, 실험적으로는 다른 결과
    - 이 논문 이전까지의 접근은 receptive field를 키워서 broad한 feature 탐색으로 성능 향상을 꾀함
        - filter size를 늘리면 parameter 수가 늘어나 training 과정에서의 어려움이 발생하는 trade-off가 있음
- 그리고 마지막의 Fully Connecter layer(FC layer)가 많아 FLOPs 크다는 단점
    - ResNet과 비교해 보면 ResNet은 layer를 많이 stack하고 FC layer는 1개만 사용함 (최신 유행)
    - VGG-Net은 layer는 적게 stack하지만 FC layer를 3개나 사용해서 FLOPs가 여기서 증가함
    - **FC layer에 관한 논문 : Overfeat**
        - Overfeat의 골자는, convolution의 경우에는 input shape-independent하지만 Fully-connected layer의 경우에는 input-shape dependent해서 다른 size의 input이 들어오면 dimension 충돌이 발생하는 문제가 있음
        - 이를 해결하기 위해서 FC-layer를 1X1 convolution으로 바꾸자(convolutionize) + dense로 해결하자는 것이 핵심
        - **FC layer의 convolutionization이 핵심**
- Scale jittering : crop vs dense (내가 만든 모델이 scale과 무관하게 scale-invariant했으면 좋겠다!)
    - crop은 현업에서 가끔씩 쓰지만, dense는 거의 안쓰고 옛날 technique이다.
    - crop는 image resize한 결과를 224X224로 자르는 방식
        - k-crop from image & its horizontal flip
        - k개의 softmax 값을 average해서 도출
    - dense는 FC ⇒ 1X1 convolution으로 바꾸고, 최종 값을 average pooling해서 1000X1X1로 바꾼 후 flatten하여 1000개 class classify한다.
    - dense + crop의 경우에는 crop의 결과와 dense의 결과를 average하는 것
- **FC ⇒ 1X1 convolutionization하는 아이디어는 semantic segmentation에서 등장함**
- Contribution of VGG
    1. 3X3 three stack instead of 7X7 filter
    2. CNN structure에서, depth를 늘리면 해결되는구나!
