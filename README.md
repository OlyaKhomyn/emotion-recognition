# Real-time emotion recognition

In this project two different emotion recognition pipelines were developed.

## Pipeline 1: based on facial landmarks and neutral face reference

Features used:
<ol>
  <li><strong>Geometric features</strong>: 68-points facial landmarks are extracted. Angles are computed using neutral image as reference.
</li>
  <li><strong>Texture-based features</strong>: Sobel filter is used to detect the edges. Density of wrinkles is calculated around 3 regions of interest.
</li>
</ol>

Neural network with 3 layers is trained to predict 7 emotions.

Example:

<img width="350" alt="Screenshot 2024-03-10 at 13 37 25" src="https://github.com/OlyaKhomyn/emotion-recognition/assets/41692593/c5120033-dc42-4799-927e-523dc8fd3126">
<img width="350" alt="Screenshot 2024-03-10 at 13 37 34" src="https://github.com/OlyaKhomyn/emotion-recognition/assets/41692593/7d5e4ac1-b6a5-498b-bd81-db4e0c8a595d">

## Pipeline 2: CNN model

