# Computer Vision Week 2. [2021 CV]

## Spatial Transformer Networks (NIPS 2015)

### References

[1] [https://kevinzakka.github.io/2017/01/10/stn-part1](https://kevinzakka.github.io/2017/01/10/stn-part1)

[2] Tensorflow KR PR12 : [Video Clip]. [https://www.youtube.com/watch?v=Rv3osRZWGbg](https://www.youtube.com/watch?v=Rv3osRZWGbg)

[3] [https://m.blog.naver.com/yl95yl/221589306754](https://m.blog.naver.com/yl95yl/221589306754)

[4] Pytorch : [Code]. [https://tutorials.pytorch.kr/intermediate/spatial_transformer_tutorial.html#id4](https://tutorials.pytorch.kr/intermediate/spatial_transformer_tutorial.html#id4)

### Introduction

<aside>
🔴 왜 spatial transformer network가 필요할까?

</aside>

- CNN은 spatial transformation에 대해 invariant하지 않기 때문이다.
    - Scaling ⇒ X
    - Rotation ⇒ X
    - Translation ⇒ △
- pooling을 통해 약간의 translation에 대해서는 invariant할 수 있다.
    - Max, Sum, Average pooling은 해당 영역 전체를 하나의 값으로 대체한다.
    - 하지만 Max-pooling layers도 receptive fields가 fixed, localized되어 있다.
- 2D Transformation의 종류
    1. Rigid Transformation : *Rotation, Translation*
    2. Similarity Transformation : Rotation, Translation + *Scaling*
    3. Affine Transformation : Rotation, Translation, Scaling + *Shearing, Reflection*
    4. Homography (Projective Transformation) : Arbitrary Square ⇒ Arbitrary Square
        
        ![Untitled](Computer%20Vision%20Week%202%20%5B2021%20CV%5D%203d0a2e1c6ce243dda744c73d00e0f0ba/Untitled.png)
        
    5. TPS(Thin Plate Spline) : Spline Interpolation의 일종, 얇은 천 위의 점들이 천이 주름짐에 따라 일그러지는 것과 같은 변환
        
        ![Untitled](Computer%20Vision%20Week%202%20%5B2021%20CV%5D%203d0a2e1c6ce243dda744c73d00e0f0ba/Untitled%201.png)
        

---

<aside>
🔴 CNN vs Spatial transformer module

</aside>

> *This limitation of CNNs is due to having only a limited, pre-defined pooling mechanism for dealing with variations in the spatial arrangement of data.*
> 

> *Spatial transformer module is a dynamic mechanism that can actively spatially transform an image or a feature map.*
> 

![변형된 MNIST Dataset에 Spatial Transformer를 적용한 결과](Computer%20Vision%20Week%202%20%5B2021%20CV%5D%203d0a2e1c6ce243dda744c73d00e0f0ba/PNG_%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5_3.png)

변형된 MNIST Dataset에 Spatial Transformer를 적용한 결과

- 일부러 찌그러뜨리거나, 노이즈를 넣은 이미지 input(a)이 spatial transformer를 거쳐 canonical한 이미지 output(c)으로 변환된다.
- (b)에서 확인할 수 있는 predicted transformation function의 형태는 input data에 따라 제각각 ⇒ dynamic mechanism이다.
- CNN 모델의 end-to-end training 과정 속에서 backpropagation을 통해 transformation function(paramerter)을 한꺼번에 학습할 수 있다.

---

### Architecture

- Spatial Transformer ⇒ Localisation network + Grid Generator + Sampler
- Spatial Transformer는 input feature map(U)를 받아 warped output feature map(V)를 반환한다.
    
    ![PNG 이미지.png](Computer%20Vision%20Week%202%20%5B2021%20CV%5D%203d0a2e1c6ce243dda744c73d00e0f0ba/PNG_%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5.png)
    

---

<aside>
🔴 Localisation network

</aside>

- Regression Layer를 마지막 단에 가지고 있어 parameter  $\theta$를 반환하는 neural network
- Fully Connected Network(FCN)이든 CNN이든 관계 없음
- input : feature map $U$, output : paratmer $\theta = f_{loc}(U)$

---

<aside>
🔴 Grid Generator

</aside>

- parameter $\theta$ 를 통해 transformation mapping $T_\theta(G)$ 생성
- $G = \{G_i\}: G_i = (x_i^t,y_i^t)$
- $\begin{bmatrix}x_i^s\\y_i^s\end{bmatrix} =T_{\theta}(G)\begin{bmatrix}x_i^t \\y_i^t \\1\end{bmatrix}$
    
    ![PNG 이미지 4.png](Computer%20Vision%20Week%202%20%5B2021%20CV%5D%203d0a2e1c6ce243dda744c73d00e0f0ba/PNG_%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5_4.png)
    

- *affine transformation*
    
    $T_{\theta}(G) = A_{\theta} = \begin{bmatrix}
    \theta_{11} & \theta_{12} & \theta_{13}\\
    \theta_{21} & \theta_{22} & \theta_{23} \end{bmatrix}$
    
    $\begin{bmatrix}x_i^s\\y_i^s\end{bmatrix} =\begin{bmatrix}
    \theta_{11} & \theta_{12} & \theta_{13}\\
    \theta_{21} & \theta_{22} & \theta_{23} \end{bmatrix}\begin{bmatrix}x_i^t \\y_i^t \\1\end{bmatrix}$
    
    ![Untitled](Computer%20Vision%20Week%202%20%5B2021%20CV%5D%203d0a2e1c6ce243dda744c73d00e0f0ba/Untitled%202.png)
    

- *attention*
    
    $T_{\theta}(G) = A_{\theta} = \begin{bmatrix}
    s & 0 & t_x\\
    0 & s & t_y \end{bmatrix}$
    
    $\begin{bmatrix}x_i^s\\y_i^s\end{bmatrix} =\begin{bmatrix}
    s & 0 & t_x\\
    0 & s & t_y \end{bmatrix}\begin{bmatrix}x_i^t \\y_i^t \\1\end{bmatrix}$
    
    ![Untitled](Computer%20Vision%20Week%202%20%5B2021%20CV%5D%203d0a2e1c6ce243dda744c73d00e0f0ba/Untitled%203.png)
    
- 여기서  $T_{\theta}$는 $U$→$V$로 가는 Transformation으로 생각할 수 있지만, 실제로 행렬 계산 시에는 Canonical → Distorted로 가는 Transformation이므로 반대 방향이다.
- Transformation   $T_{\theta}$ 의 parameter $\theta$로 affine transformation, attention 외에도 projective transformation, thin-plate-spline(TPS) tarnsformation 등을 모두 표현할 수 있다.

---

<aside>
🔴 Sampler : Differentiable Image Sampling

</aside>

- Sampler는 input feature map U로부터 sampled output feature map V를 생성한다.

![Untitled](Computer%20Vision%20Week%202%20%5B2021%20CV%5D%203d0a2e1c6ce243dda744c73d00e0f0ba/Untitled%204.png)

- output feature map V의 좌표마다, U의 어느 point에서 값을 가져올지 $T_{\theta}(G)$가 결정한다.
- 이때, V의 좌표에 대응되는 U의 point가 정수 격자점이 아닐 수도 있으므로 인접한 정수 격자점 값의 interpolation을 통해 V의 값을 가져온다.
    
    ![(3) interpolation 공식](Computer%20Vision%20Week%202%20%5B2021%20CV%5D%203d0a2e1c6ce243dda744c73d00e0f0ba/Untitled%205.png)
    
    (3) interpolation 공식
    
    ![PNG 이미지 3.png](Computer%20Vision%20Week%202%20%5B2021%20CV%5D%203d0a2e1c6ce243dda744c73d00e0f0ba/PNG_%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5_3%201.png)
    
- Interpolation은 아래에 나와 있는 두 가지 방법으로 수행 가능하다.

![(4) Nearest Integer Interpolation, (5) Bilinear Interpolation](Computer%20Vision%20Week%202%20%5B2021%20CV%5D%203d0a2e1c6ce243dda744c73d00e0f0ba/Untitled%206.png)

(4) Nearest Integer Interpolation, (5) Bilinear Interpolation

![Untitled](Computer%20Vision%20Week%202%20%5B2021%20CV%5D%203d0a2e1c6ce243dda744c73d00e0f0ba/Untitled%207.png)

![제일 가까운 격자점 값으로 대체하는 방법](Computer%20Vision%20Week%202%20%5B2021%20CV%5D%203d0a2e1c6ce243dda744c73d00e0f0ba/PNG_%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5_2.png)

제일 가까운 격자점 값으로 대체하는 방법

![점을 둘러싸는 네 개의 격자점 값의 선형 결합](Computer%20Vision%20Week%202%20%5B2021%20CV%5D%203d0a2e1c6ce243dda744c73d00e0f0ba/PNG_%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5_5.png)

점을 둘러싸는 네 개의 격자점 값의 선형 결합

- Loss 값을 Backpropagate하기 위해서는 interpolation function이 미분 가능해야 한다.
- Interpolation function이 불연속이어도 구간별로 나눠서 Backpropagation 수행할 수 있다.

![Untitled](Computer%20Vision%20Week%202%20%5B2021%20CV%5D%203d0a2e1c6ce243dda744c73d00e0f0ba/Untitled%208.png)

---

<aside>
🔴 Spatial Transformer Network

</aside>

- Spatial Transformer Network라는 하나의 모듈을 CNN 앞단에 결합해서 Image 자체를 사전에 transform하거나, CNN의 layer 사이에 끼워넣어 Convolution layer를 거친 output feature map을 transform할 수도 있다.
- 개수와 구조상의 위치는 이론적으로 제한이 없다.
- Spatial transformer가 어떻게 input feature map을 transform할 지는 CNN의 전체 cost function을 최소화하는 training 과정 중에 학습되므로, Training speed를 크게 감소시키지 않는다.
- 오히려 Attentive model에서는 downsampling 효과가 있어 speedup하기도 한다.
- STN의 여러 가지 활용에 대해 실험
    - Distorted MNIST dataset에 적용하여 성능평가
    - STN module을 CNN의 layer 사이에 여러 개 끼워넣는 경우(ST-CNN multi model)
    - 병렬적으로 여러 개 STN 배치해서 image의 다른 부분 tracking
    - MNIST addition(두 개 숫자 동시에 인식), Co-localisation(수십 개 숫자 인식) Experiment

### Experiments

<aside>
🔴 Distorted MNIST

</aside>

- MNIST handwriting dataset에 distortion을 다양한 방법으로 적용
    - *Rotation(R), Rotation/Scale/Translation(RTS), Projective Transformation(P), Elastic Warping(E)*
- Baseline FCN, CNN **vs** ST-FCN, ST-CNN(CNN : 2 max-pooling layers
- Spatial Transformer uses bilinear sampling, but different transformation functions
    - *Affine transformation(Aff), Projective transformation(Proj), 16-point thin plate spline transformation(TPS)*
- Approximately same parameters, trained with identical optimisation schemes(SGD, backprop, scheduled learning rate decrease, multinomial cross entropy loss)

![Model, Distortion 유형별 percentage error, 왼쪽은 TPS(Thin Plate Spline), 오른쪽이 Aff(Affine Transformation)](Computer%20Vision%20Week%202%20%5B2021%20CV%5D%203d0a2e1c6ce243dda744c73d00e0f0ba/PNG_%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5_6.png)

Model, Distortion 유형별 percentage error, 왼쪽은 TPS(Thin Plate Spline), 오른쪽이 Aff(Affine Transformation)

- Spatial transformer enabled network outperforms its counterpart baseline network
- CNN 계열이 FCN보다 더 나은 결과를 보이는데, 그 이유는 CNN의 max-pooling layer가 spatial invariance에 기여하고 convolutional layer가 모델링을 더 잘하기 때문이다.
- Spatial transformer를 결합한 모델이 성능 향상됨 : ST module이 Spatial invariance에 기여함
- 추가로 noisy environment(background clutter를 포함하고 있는 60X60 MNIST digits이미지)에서 FCN(13.2%), CNN(3.5%) > ST-FCN(2.0%), ST-CNN(1.7%) error를 줄임
    
    ![Noisy environment](Computer%20Vision%20Week%202%20%5B2021%20CV%5D%203d0a2e1c6ce243dda744c73d00e0f0ba/Untitled%209.png)
    
    Noisy environment
    
- TPS: most powerful, able to reduce error on elastically deformed digits(E), reduce complexity of the task, does not overfit on simpler data(예: R)
- CNN으로 classify하지 못하는 것들도 ST-CNN이 classify했다. (https://goo.gl/gdEhUu)
- 그리고 Transformation은 모두 숫자 이미지를 "standard, upright posed digit"으로 변환시키는 방향으로 학습되었다.
    - Transformation 이후 이미지로부터 classification하는 것이 최종 Task인데도, 그 과정에서 학습된 transformation은 upright pose로 변환하도록 학습되었다는 의미를 가진다.

<aside>
🔴 Street View House Numbers(SVHN)

</aside>

- SVHN Dataset은 1자리~5자리수의 집 호수 이미지(Real-world data), 20만여개 데이터로 구성
- 64X64(Tight crop), 128X128 crop : image preprocessing
- ST : affine transformation, bilinear sampling kernels
    - ST-CNN Single : 4-layer CNN으로 localisation net을 구성한 CNN spatial transformer를 CNN앞에 배치한 모델
    - ST-CNN Multi : 아래 그림의 (a)처럼 CNN의 첫 4개 convolutional layer의 앞 단에 2-layer FCN spatial transformer를 하나씩 삽입한 모델

![Maxout CNN, CNN, DRAM(Deep Recurrent Attention Model)과 ST-CNN(Single, Multi) 비교](Computer%20Vision%20Week%202%20%5B2021%20CV%5D%203d0a2e1c6ce243dda744c73d00e0f0ba/Untitled%2010.png)

Maxout CNN, CNN, DRAM(Deep Recurrent Attention Model)과 ST-CNN(Single, Multi) 비교

- Spatial transformer module이 CNN architecture 전에 결합하는 경우(ST-CNN single)와 위 그림처럼 Spatial transformer module이 CNN의 convolution layer 사이사이에 위치하는 경우(ST-CNN multi)를 비교했을 때 0.1% 감소, 유의미한 차이 없었음
    
    ![PNG 이미지 7.png](Computer%20Vision%20Week%202%20%5B2021%20CV%5D%203d0a2e1c6ce243dda744c73d00e0f0ba/PNG_%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5_7.png)
    
- ST-CNN multi에서 더 깊은 layer에 위치한 Spatial transformer의 경우에는 enriched된 feature에 대해서 transformation을 예측해야 했다.
- 이전의 SOTA model이었던 DRAM(Deep Recurrent Attention Model)과 비교했을 때 128px 이미지 input에 대해서는 4.5% → 3.9%를 달성했다.(ST-CNN single, multi)
    - DRAM은 ensemble of models + monte carlo averaging를 사용하는 반면 ST-CNN model은 single model에서 single forward pass만 하면 충분하다고 주장한다.
    - resolution & network capacity를 transformer에서 crop/rescale한 부분에만 집중하면 되고, ST-CNN Multi가 기존 CNN보다 6%만 느려졌다고 주장한다.

<aside>
🔴 Fine-Grained Classification

</aside>

- 200종의 새 사진 11,788장(6k training, 5.8k test)으로 구성된 Caltech의 CUB-200-2011 birds 데이터셋에 fine-grained bird classification을 적용한 실험
- Baseline : Inception architecture에 batch normalisation을 사용한 CNN 구조
- spatial transformer 2개(2×ST-CNN) 또는 4개(4×ST-CNN)를 병렬로 사용해서 자동으로 object에서 중요한 부분을 attention하게 학습
    - 아래 그림처럼, transformer는 각기 다른 image part를 capture하고, capture된 이미지는 Inception에 의해 initialize 되어 있는 part description sub-net으로 들어감
    - 각각 representation을 도출해서 concat ⇒ classified with single softmax layer

![PNG 이미지 8.png](Computer%20Vision%20Week%202%20%5B2021%20CV%5D%203d0a2e1c6ce243dda744c73d00e0f0ba/PNG_%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5_8.png)

![PNG 이미지 9.png](Computer%20Vision%20Week%202%20%5B2021%20CV%5D%203d0a2e1c6ce243dda744c73d00e0f0ba/PNG_%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5_9.png)

![Untitled](Computer%20Vision%20Week%202%20%5B2021%20CV%5D%203d0a2e1c6ce243dda744c73d00e0f0ba/Untitled%2011.png)

- 모든 ST-CNN이 Baseline CNN 모델보다 높은 성능을 보였다.
- red 박스는 새의 head 부분을, green 박스는 body의 중심 부분을 찾도록 별도의 supervision 없이 스스로 학습되었다(not explicitly defined, data-driven).

### Conclusion & Appendix

- Spatial Transformer ⇒ Self-contained module for neural networks
- Explicit하게 spatial transformation을 직접 수행해서 input으로 넣어주는 대신, end-to-end로 spatial transformation이 학습되도록 한다.
- Loss Function에 대한 Modification이 필요하지 않다.
- SOTA baseline CNN model보다 task 수행 accuracy 향상시켰다.
- MNIST addition(두 개 숫자 동시에 인식), Co-localisation(수십 개 숫자 인식) Experiment, 3D transformation으로 확장했다.

<aside>
🔴 MNIST Addition :  2-channel input handwriting data를 보고서 두 숫자의 합을 출력

</aside>

- 2-channel input에 대해서 병렬적으로 ST1, ST2를 학습한다.
- ST1은 아래 그림처럼 Channel 1이미지를 stabilise하고, ST2는 아래 그림처럼  Channel 2이미지를 stabilise하는 것을 확인할 수 있다.
- Spatial Transformation의 indenpendency를 확인할 수 있다.

![Untitled](Computer%20Vision%20Week%202%20%5B2021%20CV%5D%203d0a2e1c6ce243dda744c73d00e0f0ba/Untitled%2012.png)

- 2-channel Input 이미지를 활용해 독립적으로 ST1, ST2를 학습시킨 후 cross-prediction을 시켜서 4-channel output을 생성, 이것을 concatenate해서 FCN을 통해 predict한다.
    
    ![PNG 이미지 10.png](Computer%20Vision%20Week%202%20%5B2021%20CV%5D%203d0a2e1c6ce243dda744c73d00e0f0ba/PNG_%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5_10.png)
    

<aside>
🔴 Co-localisation, Higher Dimensional Transformers

</aside>

[https://www.youtube.com/watch?v=Ywv0Xi2-14Y](https://www.youtube.com/watch?v=Ywv0Xi2-14Y)

<aside>
🔴 Spatial Transformation Network 이후의 논문들

</aside>

### Deformable Convolutional Networks

[PR-002: Deformable Convolutional Networks (2017)](https://www.youtube.com/watch?v=RRwaz0fBQ0Y)