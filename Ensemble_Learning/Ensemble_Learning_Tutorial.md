# 2022 Business Analytics Chapter 4: Ensemble Learning ๐คนโโ๏ธ
## Python Tutorial: Bagging(Random Forest) vs. Boosting(Gradient Boosting & CatBoost)
### 2022010558 ๊น์งํ๐ฒ

<br/>

<br/>

# Ensemble Learning: Overview
ํ๋์ค์ด๋ก 'ensemble'์ '์กฐํ' ํน์ 'ํต์ผ'์ ์๋ฏธํฉ๋๋ค. ์ด ๋จ์ด์ ์๋ฏธ๋ฅผ ์ฐจ์ฉํ์ฌ ์ด๋ฆ ์ง์ด์ง 'Ensemble Learning'์ ์๋ฏธ๋, ๋จ์ผ ๋ชจ๋ธ๋ค์ ์ฌ๋ฌ ๊ฐ ๋ชจ์ ํ์ต์์ผ, ๊ฐ ๋ชจ๋ธ์ ์์ธก ๊ฐ์ ์ทจํฉํ๋ ๋ฐฉ์์ ๋ปํฉ๋๋ค. ๋ฐ๋ผ์ Ensemble Learning์ ์ผ๋ฐ์ ์ผ๋ก ๋จ์ผ ๋ชจ๋ธ๋ณด๋ค ๋ ๋์ ์ฑ๋ฅ์ ์๋ํ๋ค๋ ํน์ง์ ๊ฐ์ง๋๋ค.   

**๊ทธ๋ ๋ค๋ฉด Ensemble Learning์ผ๋ก ์ด๋ป๊ฒ ๋ ๋์ ์ฑ๋ฅ์ ๊ฐ์ง ์ ์๋ ๊ฑธ๊น์?**

<br/>

## <span style="color:darkblue">Bais</span>-<span style="color:purple">Variance</span> Decomposition
์ฐ๋ฆฌ๊ฐ ๋ง๋๋ ์์ธก ๋ชจ๋ธ์ ๋ฏธ๋์ ๋ค์ด์ค๋ ์๋ก์ด x ๋ฐ์ดํฐ์ ๋ํ ์์ธก ์ค์ฐจ์ ๊ธฐ๋๊ฐ์ ์ค์ด๋ ๊ฒ์ ๋ชฉํ๋ก ํ๋ฉฐ, ์ด๋ **์ค์ฐจ**์ ๊ธฐ๋๊ฐ์ ๋ชจ๋ธ์ **Bias(ํธํฅ)** ์ **Varaince(๋ถ์ฐ)** ๋ก ๋ถํด๋ฉ๋๋ค.   

<p align="center">
    <img src="Img/Noise.PNG" width="800"/>
</p>

๋ฌผ๋ก  ์ค์ฐจ์๋ Bias์ Variance๋ง ์๋ ๊ฒ์ด ์๋๋๋ค. Noise๋ผ ํ์ฌ, ๋ฐ์ดํฐ ์์ง ์์ ์์ฐ ๋ฐ์์ ์ผ๋ก ์ผ์ด๋๋ ๋ถ๊ฐํผํ ๋ณ๋๋ ์์ฃ . ์ด์  [Anomaly Detection ํํ ๋ฆฌ์ผ](https://github.com/Im-JihyunKim/BusinessAnalytics/blob/main/Anomaly_Detection/Anomaly_Detection_Tutorial.md)์์๋ ๋ค๋ฃจ์์ง๋ง, ๋ธ์ด์ฆ๋ ์ ํํ ์ถ์ ์ ๋ถ๊ฐ๋ฅํ๊ณ , ๋ค๋ง ์๋ก ๋๋ฆฝ์ ์ด๊ณ  ์ผ์ ํ ๋ถ์ฐ์ ๊ฐ์ง๋ค๊ณ  ์์๋ก ๊ฐ์ ํ๋ ๊ฐ์๋๋ค. ๊ทธ๋ฆฌ๊ณ  ์ ๋ชจ๋ฅด๋ ๊ฐ์ฐ์์ ๋ถํฌ๋ฅผ ๋ฐ๋ฅธ๋ค๊ณ  ๊ฐ์ ํ์ฃ .   

๊ทธ๋ ๋ค๋ฉด Bias์ Variance๋ ๋ฌด์์ผ๊น์? ์ ์ฌ๋ผ์ด๋์๋ ์ ํ์์ง๋ง, ๋จผ์  **<span style="color:darkblue">Bias๋ ๋ชจ๋ธ์ ๋ฐ๋ณต์ ์ผ๋ก ํ์ต์์ผฐ์ ๋ ๋์ถ๋๋ ์์ธก ๊ฐ์ ํ๊ท **์ ์๋ฏธํฉ๋๋ค. ํ๊ท ์ ์ผ๋ก ์ผ๋ง๋ ์ ํํ ์ถ์ ์ด ๊ฐ๋ฅํ์ง๋ฅผ ์ธก์ ํ๋ ์งํ์ด์ฃ . **<span style="color:purple">Variance๋ ๋ชจ๋ธ์ ๋ฐ๋ณต์ ์ผ๋ก ํ์ต์์ผฐ์ ๋ ๊ฐ๋ณ์ ์ธ ์์ธก ๊ฐ์ด ํ๊ท  ์์ธก ๊ฐ๊ณผ ์ผ๋ง๋ ์ฐจ์ด ๋๋์ง๋ฅผ ์ธก์ **ํ๋ ์งํ์๋๋ค. ์์ธก ์ถ์  ๊ฐ์ ํธ์ฐจ๋ฅผ ๊ณ์ฐํ๋ ์งํ์ธ ๊ฒ์ด์ฃ .

์ด๋ ๋ชจ๋ธ์ ์ค์ฐจ๊ฐ Bias์ Variance๋ก Decomposition ๋๋ค๋ ์๋ฏธ๋ ๋ญ๊น์? ๋ง์ผ ์ฐ๋ฆฌ์ ์์ธก Task๊ฐ Regression์ด๊ณ , ์์ธก ์ค์ฐจ๋ฅผ MSE๋ฅผ ํตํด ๊ณ์ฐํ๋ค๊ณ  ํด๋ด์๋ค. ๊ทธ๋ ๋ค๋ฉด ๋ฐ์ดํฐ $x_0$๊ฐ ๋ค์ด์์ ๋ ๋ชจ๋ธ์ Error๋ ์๋์ ๊ฐ์ต๋๋ค.

$$Expected \ MSE(x_0) = E[y-\hat{F}(x)|x=x_0]^2$$

์ด๋ ์ถ์  ๊ฐ์ ์ธ์ ๋ ๋ธ์ด์ฆ๋ฅผ ๊ฐ์ง๊ธฐ ๋๋ฌธ์, epsilon์ ์ด์ฉํด์ ์๋์ฒ๋ผ ๋ค์ ํ๊ธฐํ  ์ ์๊ฒ ์ฃ . Nosie๋ ๋๋ฆฝ์ผ๋ก ๊ฐ์ ํ๋ ๋ฐ๊นฅ์ผ๋ก ๋ค์ ๋นผ์ $\sigma ^2$๋ก ํํํฉ๋๋ค.

$$\begin{aligned}
Expected \ MSE(x_0) &= E[y-\hat{F}(x)|x=x_0]^2 \\
&= E[F^*(x_0)+ฮต - \hat{F}(x_0)]^2 \\
&= E[F^*(x_0) - \hat{F}(x_0)]^2 + \sigma ^2
\end{aligned}$$

์์ ์ธ๊ธํ Bias์ Variance์ ๋ํ ๊ฐ๋์ ์ง์ด๋ด์๋ค. ๋ ๋ชจ๋ "์์ธก ๊ฐ์ ํ๊ท  $\bar{F}(x)$"์ ์ด์ฉํ์ฌ ๊ณ์ฐ๋ฉ๋๋ค. ๊ทธ๋ ๋ค๋ฉด ์ ์์์์ ์์ธก ๊ฐ์ ํ๊ท ์ ๋ํ๊ณ  ๋นผ์ฃผ๋ฉด ์ด๋ป๊ฒ ๋ ๊น์? ๋์ผํ ๊ฐ์ ๋ํ๊ณ  ๋นผ์ฃผ๋ ์์์ ๋์ผํด์ง๊ฒ ์ฃ .

$$\begin{aligned}
Expected \ MSE(x_0) &= E[y-\hat{F}(x)|x=x_0]^2 \\
&= E[F^*(x_0)+ฮต - \hat{F}(x_0)]^2 \\
&= E[F^*(x_0) - \hat{F}(x_0)]^2 + \sigma ^2 \\
&= E[F^*(x_0) - \bar{F}(x_0) + \bar{F}(x_0) \hat{F}(x_0)]^2 + \sigma ^2
\end{aligned}$$

๋งจ ์๋ ์์์ ๋ณด๋ $(A+B)^2$ ์ ๊ผด์๋๋ค. ์ด๋ฅผ ํ์ด์ ์ ๊ฐํด๋ณด๊ฒ ์ต๋๋ค.

$$\begin{aligned}
Expected \ MSE(x_0) &= E[y-\hat{F}(x)|x=x_0]^2 \\
&= E[F^*(x_0)+ฮต - \hat{F}(x_0)]^2 \\
&= E[F^*(x_0) - \hat{F}(x_0)]^2 + \sigma ^2 \\
&= E[F^*(x_0) - \bar{F}(x_0) + \bar{F}(x_0) \hat{F}(x_0)]^2 + \sigma ^2 \\
&= E[F^*(x_0)-\bar{F}(x_0)]^2 + E[\bar{F}(x_0)-\hat{F}(x_0)]^2+ฯ^2 \\
&= \color{Purple} \color{DarkBlue} [F^*(x_0)-\bar{F}(x_0)]^2 \color{Black}+ \color{Purple} E[\bar{F}(x_0)-\hat{F}(x_0)]^2 \color{Black}+ฯ^2 \\
&=\color{DarkBlue}Bias^2(F(x_0)) \color{Black}+ \color{Purple}Var(\hat{F}(x_0)) \color{Black} +ฯ^2
\end{aligned}$$

๋จผ์  $\color{darkblue}F^*(x_0)-\bar{F}(x_0)$์ $x_0$๊ฐ ์ง์ง ์ ๋ต $F^*$์ $x_0$๊ฐ ์๋ ฅ๋์์ ๋ ๋ชจ๋ธ๋ค์ ์์ธก ํ๊ท  $\bar{F}(x_0)$ ๊ฐ์ ์ฐจ์ด๋ฅผ ๊ณ์ฐํ ์์๋๋ค. ์ฆ, ์ด๋ **<span style="color:darkblue">Bias(ํธํฅ)** ๋ฅผ ์๋ฏธํ๋ ์์ธ ๊ฒ์๋๋ค.   

$\color{purple}\bar{F}(x_0)-\hat{F}(x_0)$ ์ $x_0$๊ฐ ์๋ ฅ๋์์ ๋ ๋ชจ๋ธ๋ค์ด ์์ธกํ ๊ฐ์ ํ๊ท  $\bar{F}(x_0)$์ ๊ฐ๋ณ ์์ธก ๊ฐ $\hat{F}(x_0)$ ๊ฐ์ ์ฐจ์ด๋ฅผ ๊ณ์ฐํ ์์๋๋ค. ์ด๋ **<span style="color:purple">Variance(๋ถ์ฐ)** ์ ์๋ฏธํ๋ ์์ด ๋๊ฒ ์ฃ .

๊ฒฐ๋ก ์ ์ผ๋ก ์์์ ๋ฏธ๋ ๋ฐ์ดํฐ $x_0$์ ์ค์ฐจ ๊ธฐ๋๊ฐ์, ๋ชจ๋ธ์ **<span style="color:darkblue">Bias(ํธํฅ)** ์ **<span style="color:purple">Variance(๋ถ์ฐ)**, ๊ทธ๋ฆฌ๊ณ  **Natural Error์ธ Noise** ์ด 3๊ฐ์ง ์์๋ก ๋ถํดํ  ์ ์์ต๋๋ค. ๊ทธ๋ฆฌ๊ณ  Noise๋ ์์ฐ ๋ฐ์์ ์ด๊ณ  ๋ถ๊ฐํผํ ๋ณ๋์ด๋, ์ฐ๋ฆฌ๊ฐ ์ค์ฌ๋๊ฐ ์ ์๋ ๊ฒ์ Bias์ Variance๊ฐ ๋๊ฒ ์ฃ .   

### <span style="color:darkblue">Bias</span>์ <span style="color:purple">Variance</span>์ ๋ฐ๋ฅธ ๋ชจ๋ธ ๊ตฌ๋ถ
๊ทธ๋ ๋ค๋ฉด ์์ธก ๋ชจ๋ธ์ ์์์ ๋งํ ์ค๋ฅ์ ๋ฐ๋ผ, Bias๊ฐ ๋ฎ์ ๋ชจ๋ธ๊ณผ Variance๊ฐ ๋ฎ์ ๋ชจ๋ธ๋ก ๊ตฌ๋ถํ  ์ ์์ ๊ฒ์๋๋ค.   

**<span style="color:darkblue">๋ชจ๋ธ์ Bias**๊ฐ ํฌ๊ณ  ์๋ค๋ ๊ฒ์ ๊ฒฐ๊ตญ ๋ฌด์จ ์๋ฏธ์ผ๊น์?
- **<span style="color:darkblue">๋ชจ๋ธ์ Bias๊ฐ ํฌ๋ค๋ฉด**, ์ฐ๋ฆฌ์ ํ์ต ๋ฐ์ดํฐ์ ํจํด์ ์ ๋๋ก ๋ฐ์ํ์ง ๋ชปํ๋ค๋ ์๋ฏธ, ์ฆ Training Error๊ฐ ํฌ๋ค๋ ์๋ฏธ์ผ ๊ฒ์๋๋ค.
- **<span style="color:darkblue">๋ชจ๋ธ์ Bias๊ฐ ๋ฎ๋ค๋ฉด**, ํ์ต ๋ฐ์ดํฐ์ ํจํด์ ์ ๋๋ก ๋ฐ์ํ์ฌ์ Training Error๋ฅผ ์ต์ํ ํ๋ค๋ ์๋ฏธ์ด๊ฒ ์ฃ . ์ค์  ๊ฐ๊ณผ ์์ธก ๊ฐ ํ๊ท ์ ์ฐจ์ด๊ฐ ์๋ค๋ ๊ฒ์ด๋๊น์.

**<span style="color:purple">๋ชจ๋ธ์ Variance**๊ฐ ํฌ๊ณ  ์๋ค๋ ๊ฒ์ ๋ค์๊ณผ ๊ฐ์ ์๋ฏธ๋ฅผ ๊ฐ์ง๋๋ค.
- **<span style="color:purple">๋ชจ๋ธ์ Variance๊ฐ ํฌ๋ค๋ฉด**, ๋ฐ์ดํฐ๊ฐ ๋ฐ๋๋ค๋ฉด, ๋ชจ๋ธ์ ์์ธก ๊ฐ์ ๋ง์ ๋ณ๋์ด ์์๋๋ค๋ ๊ฒ์๋๋ค.
- **<span style="color:purple">๋ชจ๋ธ์ Variance๊ฐ ๋ฎ๋ค๋ฉด**, ๋ฐ์ดํฐ ๋ฐ ํ์ดํผํ๋ผ๋ฏธํฐ๊ฐ ๋ฐ๋๋ค ํ๋๋ผ๋, ์์ธก ๊ฐ์ ๋ณ๋์ด ํฌ์ง ์๋ค๋ ์๋ฏธ์๋๋ค. Variance๋ ๋ชจ๋ธ์ ์์ธก ๊ฐ์ด ํ๊ท  ์์ธก ๊ฐ์ผ๋ก๋ถํฐ ํผ์ง ์ ๋๋ฅผ ์ธก์ ํ๋ ๊ฒ์ด๋, ์์ธก ๊ฐ ์ฌ์ด์ ํฐ ์ฐจ์ด๊ฐ ์๋ค๋ ๊ฒ์ด์ฃ .

์ด๋ฅผ ํตํด์ ๋ชจ๋ธ์ ์๋์ ๊ฐ์ด ์ด 4๊ฐ์ง ๊ฒฝ์ฐ๋ก ๊ตฌ๋ถํ  ์ ์์ ๊ฒ์๋๋ค.
<p align="center">
    <img src="Img/Bias_Variance.PNG" width="800"/>
</p>

Case 1์ Bias์ Variance๊ฐ ๋ชจ๋ ๋์ ๊ฒฝ์ฐ๋ก, ์ฑ๋ฅ์ด ๋งค์ฐ ์ข์ง ์์ Worst Case์๋๋ค. **์ฐ๋ฆฌ๊ฐ ๋ค๋ฃจ๋ ๋๋ถ๋ถ์ ์์ธก ๋ชจ๋ธ์ Case 2์ Case 3๋ก ๊ตฌ๋ถ**ํ  ์ ์์ฃ .
- **<span style="color:darkblue">Case 2๋ Bias๊ฐ ๋ฎ๊ณ  Variance๊ฐ ๋์ ๋ชจ๋ธ**์๋๋ค.
    - Bias๊ฐ ๋ฎ๊ธฐ ๋๋ฌธ์ Training Error๊ฐ Case 3๋ณด๋ค ๋ฎ์ง๋ง, Variance๊ฐ ๋๊ณ  ๊ตฌ๊ฐ ์ถ์  ๋ฒ์๊ฐ ๋์ด Testing Error๊ฐ ๋๋ค๋ ํน์ง์ ๊ฐ์ง๋๋ค. ์ฆ **Overfitting**์ ๊ฒฝํฅ์ ๋ณด์ผ ์ ์๋ ๋ชจ๋ธ์ด์ฃ .
    - ์ด๋ Case 2์ ์ํ๋ ๋ชจ๋ธ์ ๊ธฐ๋ณธ์ ์ผ๋ก **๋ชจ๋ธ์ ๋ณต์ก๋(VC Dimension)๊ฐ ๋์ ๊ฐ๋ณ์ ์ธ Training Error(Empirical Error)๊ฐ ๋ฎ์ต๋๋ค**.   

<br/>

- **<span style="color:purple">Case 3์ Variance๊ฐ ๋ฎ๊ณ  Bias๊ฐ ๋์ ๋ชจ๋ธ**์๋๋ค.
    - Variance๊ฐ ๋ฎ๊ธฐ ๋๋ฌธ์ Training Error๋ ๋๊ณ , Testing์ ๋ํ ์์ธก๋ ฅ๋ ๋ฎ์ฃ . ์ด๋ **Underfitting**์ด ๋ ์ ํ์ ์ธ ๋ชจ๋ธ์ ์๋ผ๊ณ  ํ  ์ ์์ต๋๋ค.
    - Case 3์ ์ํ๋ ๋ชจ๋ธ์ ๊ธฐ๋ณธ์ ์ผ๋ก **๋ชจ๋ธ์ ๋ณต์ก๋(VC Dimension)๊ฐ ๋ฎ๋ค๋ ํน์ง**์ ๊ฐ์ง๊ณ  ์์ต๋๋ค.

> <span style="color:gray">__[์ฐธ๊ณ ]__   
VC Dimension๊ณผ Empirical Error์ ๊ด๊ณ์ ๋ํด์๋ [Kernel Based Learning ํํ ๋ฆฌ์ผ](https://github.com/Im-JihyunKim/BusinessAnalytics/blob/main/Ch2_Kernel_Based_Learning(SVM)_Tutorial.ipynb)์ ๋ณด๋ค ์์ธํ ๊ธฐ์ ๋์ด ์์ต๋๋ค.

<br/>

## <span style="color:darkblue">Bagging</span> vs. <span style="color:purple">Boosting
๊ทธ๋ ๋ค๋ฉด Ensemble Learning์์๋ ์ด๋ป๊ฒ ๋จ์ผ๋ชจ๋ธ๋ณด๋ค ์์ธก ์ค๋ฅ๋ฅผ ๊ฐ์์ํฌ ์ ์์๊น์?   
๋ณธ ํํ ๋ฆฌ์ผ์์ ๋ค๋ฃฐ Ensemble Learning์ ๋ฐฉ๋ฒ๋ก ์ **<span style="color:darkblue">Bagging**๊ณผ **<span style="color:purple">Boosting**์ด๋ฉฐ, ๊ฐ ๋ฐฉ๋ฒ๋ก ์ ๋ค์๊ณผ ๊ฐ์ ํน์ง์ ๊ฐ์ง๋๋ค.   

<p align="center">
    <img src="Img/bagging_boosting_1.PNG" width="800"/>
</p>

**<span style="color:darkblue">Bagging (Bootstrap Aggregating)</span>**:   
- **<span style="color:darkblue">ํ์ต ๋ฐ์ดํฐ์์ Randomํ๊ฒ ์ถ์ถํ์ฌ ๋ชจ๋ธ์ ๊ฐ๊ฐ ๋ค๋ฅด๊ฒ ํ์ต์ํค๋ ๋ฐฉ๋ฒ๋ก **์๋๋ค.
- ๋ฐ๋ผ์ **<span style="color:darkblue">๊ฐ๋ณ ๋ชจ๋ธ๋ค์ ์๋ก ๋๋ฆฝ์ **์ด๋ฉฐ ์ํฅ์ ์ฃผ๊ณ  ๋ฐ์ง ์๋๋ค๋ ํน์ง์ด ์์ต๋๋ค. ์ฆ, ๊ฐ ๋ชจ๋ธ๋ค์ ํ์ต ๋ฐ ์ถ๋ก ์ด **<span style="color:darkblue">๋ณ๋ ฌ์ (Parallel)** ์ผ๋ก ์ด๋ฃจ์ด์ง๋ ๊ฒ์ด์ฃ .
- ์ด๋ ๋ฐ์ดํฐ๋ฅผ ๋ฌด์์๋ก ์ถ์ถํ  ๋, **์ค๋ณต์ ํ์ฉํ๋ ๋ณต์ ์ถ์ถ์ด๋ฉด Bootstrapping**, ์ค๋ณต์ ํ์ฉํ์ง ์์ผ๋ฉด Pasting์ด๋ผ ํฉ๋๋ค.
- ์ผ๋ฐ์ ์ผ๋ก๋ **<span style="color:darkblue">Bootstrapping์ ํตํด ๋ชจ๋ธ ๋ณ๋ก ํ์ต ๋ฐ์ดํฐ๋ฅผ ๋ง๋ค๊ณ , ์ด๋ค์ aggregationํ์ฌ ์์ธก ๊ฐ์ ๋ชจ์ผ๋ ๋ฐฉ์**์ ํํฉ๋๋ค. ๊ทธ๋์ "Bootstrap Aggregating"์ด๋ผ ์ด๋ฆ ๋ถ์ฌ์ง ๊ฒ์ด์ฃ .

**<span style="color:purple">Boosting**:   
- Boosting์ **<span style="color:purple">์ฑ๋ฅ์ด ์ฝํ Weak Learner๋ฅผ ์ฌ๋ฌ ๊ฐ ์ฐ๊ฒฐํ์ฌ Strong Learner๋ฅผ ๋ง๋๋ ๋ฐฉ๋ฒ๋ก **์๋๋ค.
- ์์์ ํ์ต๋ ๋ชจ๋ธ์ ์ฝ์ ์ ๋ณด์ํด ๋๊ฐ๋ฉด์ ๋ ๋์ ๋ชจ๋ธ๋ก ํ์ต์ํค๋ ๋ฐฉ์์ด์ฃ . ์ด๋ฅผํ๋ฉด **<span style="color:purple">์ด์  ๋ชจ๋ธ์ด ์๋ชป ์์ธกํ ๋ฐ์ดํฐ์ ๊ฐ์ค์น๋ฅผ ๋ถ์ฌํ๊ณ , ๋ค์ ๋ชจ๋ธ์ด ์ด์ ๋ํ ์ค๋ฅ๋ฅผ ๊ฐ์ ํด ๋๊ฐ๋ฉฐ ํ์ต ํ๋ ๋ฐฉ์**์๋๋ค.
- ๋ฐ๋ผ์ **<span style="color:purple">์ ํ ๋ชจ๋ธ์ ์ฑ๊ณผ์ ์์กด์ **์ด๋ฉฐ, ์ ํ ๋ชจ๋ธ์ ๊ฐ์ด๋๊ฐ ํ์ํ๊ธฐ์ ํ์ต ๋ฐ ์ถ๋ก ์ด **<span style="color:purple">์์ฐจ์ (Sequential)** ์ด๋ผ๋ ํน์ง์ ๊ฐ์ง๊ณ  ์์ต๋๋ค.

<br/>


<p align="center">
    <img src="Img/bagging_boosting.PNG" width="800"/>
</p>

**<span style="color:darkblue">Bagging์ Case 2์ ๊ฐ์ด Bias๊ฐ ๋ฎ์ ๋ชจ๋ธ๋ค์ ์ด์ฉํด์ Variance๋ฅผ ์ค์ฌ๋๊ฐ๋ ๋ฐฉ์์ผ๋ก ์์ธก ์ค๋ฅ๋ฅผ ๊ฐ์**์ํค๋ ๋ฐฉ๋ฒ์๋๋ค. ๊ฐ ๊ฐ๋ณ ๋ชจ๋ธ์ ์ฑ๋ฅ์ ์ข์ง๋ง, ๊ทธ ํธ์ฐจ๊ฐ ์๋ค๋ฉด ์ด๋ฅผ ์ค์ผ ์ ์๋ ๋ฐฉ๋ฒ๋ก ์ด์ฃ .   
- Case 2์ ์ํฉ์์ ์์๋ธ์ ์ฌ์ฉํ์ง ์์์ ๋ **<span style="color:darkblue">Overfitting์ด ๋ฌธ์ ๊ฐ ๋๋ค๋ฉด, ์ด๋ฅผ ํด๊ฒฐํ๋ ๋ฐ Bagging ๋ฐฉ์์ ์ด์ฉ**ํ  ์ ์๋ ๊ฒ์๋๋ค.

<br/>

**<span style="color:purple">Boosting์ Case 3๊ณผ ๊ฐ์ด Variance๊ฐ ๋ฎ์ ๋ชจ๋ธ๋ค์ ํฉ์ณ์ Bias๋ฅผ ์ค์ด๋ ๋ฐฉ์์ผ๋ก ์์ธก ์ค๋ฅ๋ฅผ ๊ฐ์์ํค๋ ๋ฐฉ๋ฒ**์๋๋ค. Sequentialํ๊ฒ ๋ชจ๋ธ์ ์ฝ์ ๋ค์ ๋ณด์ํด๋๊ฐ๋ ๊ฒ์ด์ฃ .   
- ์ฆ, ์์ธก ์ฑ๋ฅ์ ํฅ์์ ๊พํ๋ ๋ฐฉ์์ด๊ธฐ ๋๋ฌธ์, **<span style="color:purple">์์๋ธ ๋ชจ๋ธ์ ์ฌ์ฉํ์ง ์์ ๋ ์์ธก ์ฑ๋ฅ์ด ๋ฌธ์  ๋๋ฉด Boosting ๋ฐฉ์์ ํตํด ์ฑ๋ฅ์ ๋์ด๋ ๊ฒ**์ด ์ผ๋ฐ์ ์๋๋ค.

<br/>
์ ์ด์  Ensemble Learning์ ์ด๋ก ์  ๋ฐฐ๊ฒฝ๋ฟ ์๋๋ผ Bagging๊ณผ Boosting์ ์ฐจ์ด๋ฅผ ์์ ๋ณด์์ผ๋, ๋ณธ๊ฒฉ์ ์ผ๋ก ๊ตฌ์ฒด์ ์ธ ์๊ณ ๋ฆฌ์ฆ์ ๋ํ ์ค๋ช ๋ฐ ์ฝ๋๋ฅผ ํตํด ์ดํด๋ฅผ ๋์ฌ๋ณด๋๋ก ํ๊ฒ ์ต๋๋ค.

<br/>

# Set up for Python Tutorial
```python
# Import Libraries
import time
import numpy as np
np.random.seed(2022)
import pandas as pd

# For visualiation
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("talk")
sns.set_style("white")
sns.set_palette("Pastel1")
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['axes.unicode_minus'] = False

from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

import warnings
warnings.filterwarnings('ignore')
```

<br/>

# <span style="color:darkblue">Bagging 1: Bagging with Decision Tree
`Scikit-Learn`์์๋ Bagging์ ๊ฐํธํ๊ฒ ์ฌ์ฉํ  ์ ์๋ ๋ผ์ด๋ธ๋ฌ๋ฆฌ๋ฅผ ์ ๊ณตํ๊ณ  ์์ต๋๋ค. ํ๊ณ ์ ํ๋ task๊ฐ Classification์ธ ๊ฒฝ์ฐ์๋ `BaggingClassifier`๋ฅผ, Regression์ธ ๊ฒฝ์ฐ์๋ `BaggingRegressor`๋ฅผ ์ ๊ณตํ์ฃ .   

๋ณธ ํํ ๋ฆฌ์ผ์์๋ Classification ๋ฌธ์ ๋ฅผ ํ๋ฉฐ ๊ฐ ์๊ณ ๋ฆฌ์ฆ์ ํน์ง์ ์ดํด๋ณด๊ฒ ์ต๋๋ค.
```python
from sklearn.ensemble import BaggingClassifier
```
๋ฐ์ดํฐ์์ผ๋ก๋ [make_classification() ํจ์](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html)๋ฅผ ์ด์ฉํ์ฌ Binary Classification Task๋ฅผ ํ๊ธฐ ์ํ ๊ฐ์ ๋ฐ์ดํฐ๋ฅผ ๋ง๋ค์ด๋๋๋ค. ํด๋น ๋ฐ์ดํฐ์์๋ 20๊ฐ์ ์ค๋ช ๋ณ์์ 1,000๊ฐ์ ๊ด์ธก์น๊ฐ ํฌํจ๋์ด ์๋๋ก ์ค์ ํ์์ต๋๋ค.   

```python
# Define and Get Dataset
def get_dataset():
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                               n_redundant=5, random_state=2022)
    return X, y
```

<br/>

## <span style="background-color:#fff5b1"> [์คํ 1] Decision Tree vs. Bagging with Decision Tree
์์ VC Dimension์ด ๋์ ๋จ์ผ ๋ชจ๋ธ๋ก ๊ณผ์ ํฉ์ด ๋ฐ์ํ๋ ๊ฒฝํฅ์ด ์์ ๋, Bagging์ ํตํด์ ์ด๋ฅผ ์ํํ  ์ ์๋ค๊ณ  ํ์์ต๋๋ค. ๊ทธ ๋ํ์ ์ธ Base Learner๋ก๋ (1) Decision Tree, (2) Artificial Neural Network, (3) k-NN with small K ๋ฑ์ด ์์ต๋๋ค. ์ด๋ฌํ ๋ชจ๋ธ์ ๋ํ Variance๊ฐ ๋์ ์ค๋ฅ๋ฅผ ๋ฐ์์ํค๋ ํน์ง์ ๊ฐ์ง๋๋ฐ, **<span style="color:maroon">๊ณผ์ฐ Bagging์ ํตํด์ ์ฑ๋ฅ์ ํฅ์์ํฌ ์ ์์๊น์?**   

**<span style="color:maroon">๋จผ์  Decision Tree๋ฅผ ํ์ฉํ์ฌ, ๋จ์ผ ๋ชจ๋ธ์ ์ฑ๋ฅ๊ณผ Bagging์ ์ฑ๋ฅ ์ฐจ์ด๋ฅผ ๋น๊ต**ํด๋ณด๊ฒ ์ต๋๋ค. ์ด๋ **Bagging์ ์ฌ๋ฌ Decision Tree์ ์์ธก ๊ฒฐ๊ณผ๋ฅผ ์ทจํฉํ๋ ๋ฐ ์์ด์๋, ๊ฐ ๋ชจ๋ธ์ Accuracy Score์ AUROC Score์ ํ๊ท  ๊ฐ์ ์ด์ฉ**ํฉ๋๋ค.

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
```
```python
# Define Data
X, y = get_dataset()

# Single Decision Tree
tree_clf = DecisionTreeClassifier(random_state=2022)

# Bagging with Decision Tree
bag_clf_tree = BaggingClassifier(
    DecisionTreeClassifier(random_state=2022), n_estimators=500, bootstrap=True, n_jobs=-1, random_state=2022
)
```
- ๋จผ์  ์์ ๊ฐ์ด Dataset๊ณผ Model์ ์ ์ํฉ๋๋ค.
```python
def evaluate_model(model, X, y):
    # Define the evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=2022)
    # Evaluate model and collect the results
    acc = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    roc_auc = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    
    # Report Performance
    print("Accuracy: %.2f (%.2f)" % (np.mean(acc), np.std(acc)))
    print("AUROC: %.2f (%.2f)" % (np.mean(roc_auc), np.std(roc_auc)))
```
- **<span style="color:maroon">๋ชจ๋ธ์ ๊ฒฐ๊ณผ ๊ฐ์ Cross Valitaion์ ํตํด ๋ฝ์๋๋๋ค.** 
- ์ด๋ `RepeatedStratifiedKFold`๋ ๊ต์ฐจ ๊ฒ์ฆ์ ๋ฐ๋ณต์ ์ผ๋ก ์ฌ๋ฌ ๋ฒ ์ํํ  ์ ์๋ Class์๋๋ค. ์ฌ์ฉ์๊ฐ ์ง์ ํ ํ์ (`n_repeats`) ๋งํผ ๋ฐ๋ณตํด์ Fold๋ฅผ ๋๋๊ณ , Cross Validation์ ๋ํ Score๋ ๋ฐ๋ณต ํ์๋งํผ ์ป์ ์ ์์ต๋๋ค. `n_repeats`์ ๊ธฐ๋ณธ ๊ฐ์ 10์ด๋ฉฐ ๋ณธ ํํ ๋ฆฌ์ผ์์๋ 3์ ์ฌ์ฉํ์์ต๋๋ค.
- ์ด๋ ๋จธ์ ๋ฌ๋ ์๊ณ ๋ฆฌ์ฆ ๋ฐ Evaluation Procedure์ Stochastic Nature์ ์ํด์, ๊ฒฐ๊ณผ ๊ฐ์ ๊ทธ๋ ๊ทธ๋ ๋ฌ๋ผ์ง ์ ์์ต๋๋ค. ์ด๋ฅผ ๊ฐ์ํ์ฌ ํ์คํธ์ฐจ๋ฅผ ํจ๊ป ์ฐ์ถํฉ๋๋ค.

```python
# Decision Tree
evaluate_model(model=tree_clf, X=X, y=y)

# Bagging with Decision Tree
evaluate_model(model=bag_clf_tree, X=X, y=y)
```

### Results
|                 |__Decision Tree__|__Bagging with Decision Tree__|  
|-----------------|:-----------------:|:------------------:|
|__Mean Accuracy (std)__| 0.82 (0.04) | __0.90 (0.03)__ |
|__Mean AUROC (std)__| 0.82 (0.04) | __0.97 (0.01)__ |

<br/>

### <span style="background-color:#fff5b1"> [์คํ 1] ๊ฒฐ๊ณผ ํด์  
์์์ ํ์ธํ  ์ ์๋ฏ์ด, **<span style="color:maroon">Bagging์ ์ด์ฉํ ๋ฐฉ๋ฒ๋ก ์ด Accuracy ์ธก๋ฉด์์๋ 8%, AUROC ์ธก๋ฉด์์๋ 15%๋ ๋ ์ฐ์ํ๊ฒ ๋์จ ๊ฒ์ ํ์ธ**ํ  ์ ์์ต๋๋ค. Classification Task์์ Decision Tree๋ ์ด๋ป๊ฒ๋  ๋ฐ์ดํฐ ์ํ์ ํ๋์ Class๋ก ํ ๋นํ๊ธฐ ์ํด ๊ณ์ํด์ ๋ถ๊ธฐํด๋๊ฐ๋ ์ฑ์ง์ด ์๊ณ , ๋ฐ๋ผ์ ๊ณผ์ ํฉ์ ์ฐ๋ ค๊ฐ ์๋ ์๊ณ ๋ฆฌ์ฆ์๋๋ค. ๋ฐ๋ผ์ **<span style="color:maroon">Bagging์ ์ฌ์ฉํ๋ฉด ๋ณด๋ค ์ผ๋ฐํ ์ฑ๋ฅ์ด ์ข์ Decision boundary๋ฅผ ์ป๋ ๋์์, Varaince๋ฅผ ๋ฎ์ถฐ ์์ธก ์ค๋ฅ๋ฅผ ๊ฐ์**์ํฌ ์๋ ์๋ ๊ฒ์ด์ฃ .

---

<br/>

## <span style="background-color:#fff5b1"> [์คํ 2] Change Base Learner in Bagging (k-NN with small k)
์์  ์คํ์ ํตํด์ Ensemble Learning์ ์ฐ์์ฑ์ ํ์ธํ  ์ ์์์ต๋๋ค. ์ด๋ Decision Tree๋ Bagging์ Base Learner๋ก์ ์ฃผ๋ก ํ์ฉ๋๋ ์๊ณ ๋ฆฌ์ฆ์๋๋ค. ์ด๋ Variance๊ฐ ํฌ๋๋ก ๊ตฌ์ฑํ๊ธฐ๊ฐ ์ฝ๊ณ , ์ผ๋ฐ์ ์ผ๋ก Bias๊ฐ ๋ฎ๊ธฐ ๋๋ฌธ์๋๋ค. **<span style="color:maroon">k๊ฐ์ด ๋ฎ์ k-NN ์ญ์ ๋ง์ฐฌ๊ฐ์ง๋ก Variance๊ฐ ๋๊ณ  Bias๊ฐ ๋ฎ์ ๋ํ์ ์ธ ์**์๋๋ค. ๊ทธ๋ ๋ค๋ฉด ๋ง์ผ **<span style="color:maroon">Base Learner๋ฅผ ๋ฐ๊พธ์์ ๋๋ Bagging์ ํจ๊ณผ๋ฅผ ๋ณผ ์ ์์๊น์?** ์คํ์ ํตํด์ ํ์ธํด๋ณด๋๋ก ํ๊ฒ ์ต๋๋ค.

```python
from sklearn.neighbors import KNeighborsClassifier
```
```python
# Single kNN with k=5


# Bagging with kNN with k=5
bag_clf_kNN = BaggingClassifier(base_estimator=KNeighborsClassifier())
```
```python
# Evaluate 
evaluate_model(model=bag_clf_kNN, X=X, y=y)
```
- `BaggingClassifier` Class์์ Base Learner๋ฅผ ๋ฐ๊พธ๋ ค๋ฉด, `base_estimator` ์ธ์๋ฅผ `KNeighborsClassifier()`๋ก ๋ฐ๊พธ์ด์ฃผ๋ฉด ๋ฉ๋๋ค. ์ด๋ **k๊ฐ์ default๋ 5์ด๋ฉฐ, ๋ณธ ํํ ๋ฆฌ์ผ์์๋ k=3์ ์ฌ์ฉ**ํ์์ต๋๋ค.

### Results
|      |__Decision Tree__|__Bagging with DT__| __kNN(k=3)__ | __Bagging with kNN(k=3)__ |
|------|:---------------:|:----------------:|:----------------:|:------------------:|
|__Mean Accuracy (std)__| 0.82 (0.04) | 0.90 (0.03) | __0.93 (0.03)__ | __0.93 (0.03)__ |
|__Mean AUROC (std)__| 0.82 (0.04) | 0.97 (0.01) | 0.96 (0.02) | __0.97 (0.01)__ |

<br/>

### <span style="background-color:#fff5b1"> [์คํ 2] ๊ฒฐ๊ณผ ํด์
kNN์ ๊ฒฝ์ฐ, AUROC Score ์ธก๋ฉด์์๋ Bagging ๋ฐฉ์์ด 1% ๊ฐ๋ ๋์ ์ฑ๋ฅ์ ๋ณด์ด๋ ๊ฒ์ ํ์ธํ์ต๋๋ค. **<span style="color:maroon"> ๊ทธ๋ ๋ค๋ฉด ๋ค๋ฅธ k๊ฐ์ ๋ฐ๋ผ์๋ ์ฑ๋ฅ์ด ๋ฌ๋ผ์ง๊น์? 'k๊ฐ์ด ์๋ค'๋ ๊ธฐ์ค์ ๋ฌด์์ผ๊น์?** ์ด๋ [์คํ 3]์์ ํ์ธํด๋ณด๊ฒ ์ต๋๋ค.

<br/>

## <span style="background-color:#fff5b1"> [์คํ 3] Bagging with kNN with k values from 1 to 20
```python
def get_models():
    base_learner, bagging = dict(), dict()
    # Evaluate k values from 1 to 20
    for i in range(1, 20+1):
        # Define Base learner
        base_learner[str(i)] = KNeighborsClassifier(n_neighbors=i)
        # Define Ensemble Model
        bagging[str(i)] = BaggingClassifier(base_estimator=base_learner[str(i)])
    
    return base_learner, bagging
```
```python
kNN, bag_kNN = get_models()
```
- ๋จผ์  Single kNN๊ณผ kNN์ ๊ธฐ๋ณธ ๋ฒ ์ด์ค๋ก Bagging ๋ฐฉ์์ Ensemble Learning์ ์ํํ๋ ๋ชจ๋ธ์ `get_models()`๋ฅผ ํตํด ์ ์ํฉ๋๋ค. ์ด๋ k๊ฐ์ 1๋ถํฐ 20๊น์ง ๋ค๋ฅด๊ฒ ํ์ฌ ๊ฐ ๋ชจ๋ธ๋ค์ dict ์์ ๋ฃ์ด ๋ถ๋ฌ์ฌ ๊ฒ์๋๋ค.
```python
# Evaluate the models and store results

def print_results(models: dict, X, y):
    acc_score, auroc_score, k_list = [], [], []
    for k, model in models.items():
        # Evaluate Model
        acc, auroc = evaluate_model(model, X, y)
        # Store the Results
        acc_score.append(acc)
        auroc_score.append(auroc)
        k_list.append(k)
        
        # Print the performance along the way
        print('k=%s, Accuracy: %.3f (%.3f)' % (k, np.mean(acc), np.std(acc)))
        print('k=%s, AUROC: %.3f (%.3f)' % (k, np.mean(auroc), np.std(auroc)))
    
    return k_list, acc_score, auroc_score
```
```python
kNN_k_list, kNN_acc_list, kNN_auroc_list = print_results(kNN, X, y)
```
```
k=1, Accuracy: 0.915 (0.026)
k=1, AUROC: 0.915 (0.026)
k=2, Accuracy: 0.896 (0.029)
k=2, AUROC: 0.951 (0.021)
k=3, Accuracy: 0.931 (0.026)
k=3, AUROC: 0.962 (0.016)
...
k=20, Accuracy: 0.922 (0.025)
k=20, AUROC: 0.979 (0.010)
```
- ๋จผ์  Single kNN์์ k์ ๊ฐ์ด 1๋ถํฐ 20๊น์ง ๋ณํํ  ๋, ๊ฐ ๋ชจ๋ธ์ Accuracy์ AUROC๋ฅผ ์ฐ์ถํฉ๋๋ค.
```python
bag_kNN_k_list, bag_kNN_acc_list, bag_kNN_auroc_list = print_results(bag_kNN, X, y)
```
```
k=1, Accuracy: 0.915 (0.028)
k=1, AUROC: 0.960 (0.019)
k=2, Accuracy: 0.922 (0.025)
k=2, AUROC: 0.971 (0.013)
k=3, Accuracy: 0.927 (0.027)
k=3, AUROC: 0.975 (0.013)
...
k=20, Accuracy: 0.924 (0.026)
k=20, AUROC: 0.979 (0.011)
```
- ๋ค์์ผ๋ก๋ Bagging with kNN์ผ๋ก k ๊ฐ์ด 1๋ถํฐ 20๊น์ง ๋ณํํ  ๋ ๊ฐ Bagging ๋ชจ๋ธ์  Accuracy์ AUROC๋ฅผ ์ฐ์ถํฉ๋๋ค.

<br/>

### Results
### <span style="color:maroon">1. Single kNN
```python
plt.figure(figsize=(10, 5))
plt.title("Single kNN num of neighbors vs. Classification accuracy")
plt.boxplot(x=kNN_acc_list, labels=kNN_k_list, showmeans=True);
```
<p align="center">
    <img src="Img/knn_k_acc.png" width="500"/>
</p>

```python
plt.figure(figsize=(10, 5))
plt.title("Single kNN num of neighbors vs. Classification AUROC")
plt.boxplot(x=kNN_auroc_list, labels=kNN_k_list, showmeans=True);
```
<p align="center">
    <img src="Img/knn_k_auroc.png" width="500"/>
</p>

### <span style="color:maroon"> 2. Bagging with kNN
```python
plt.figure(figsize=(10, 5))
plt.title("Bagging kNN num of neighbors vs. Classification accuracy")
plt.boxplot(x=bag_kNN_acc_list, labels=bag_kNN_k_list, showmeans=True);
```
<p align="center">
    <img src="Img/bag_knn_k.png" width="500"/>
</p>

```python
plt.figure(figsize=(10, 5))
plt.title("Bagging kNN num of neighbors vs. Classification AUROC")
plt.boxplot(x=bag_kNN_auroc_list, labels=bag_kNN_k_list, showmeans=True);
```
<p align="center">
    <img src="Img/bag_knn_k_auroc.png" width="500"/>
</p>

<br/>

### <span style="background-color:#fff5b1"> [์คํ 3] ๊ฒฐ๊ณผ ํด์
- **<span style="color:maroon">Accuracy๋ฅผ ๊ธฐ์ค์ผ๋ก, Bagging์ ๋ณด๋ฉด, k๊ฐ์ด 7๋ณด๋ค ์์ ๋๋ Accuracy๊ฐ ์ฆ๊ฐํ๋ค๊ฐ, k ๊ฐ์ด ์ปค์ง ์๋ก ์คํ๋ ค ์ฑ๋ฅ์ด ํ๋ฝํ๊ฑฐ๋ ๋ณ๋์ฑ์ด ํฐ ๊ฒฝํฅ**์ ์๊ฐ์ ์ผ๋ก ๋ณผ ์ ์์์ต๋๋ค.
    - **<span style="color:maroon">ํนํ k๊ฐ 10์ ๋์ด๊ฐ๋ฉด, ์คํ๋ ค Single kNN์์ ์ข ๋ ์์ ๋ ์ฑ๋ฅ์ ํ์ธ**ํ  ์ ์์ต๋๋ค.
    - ์ด๋ k๊ฐ์ด ์ปค์ง ์๋ก Variance๊ฐ ์์ ๋ชจ๋ธ์ด ๋๊ธฐ ๋๋ฌธ์๋๋ค.
- **<span style="color:maroon">AUROC ๊ธฐ์ค์ผ๋ก๋ Single kNN์ k๊ฐ 7์ด ๋์ด๊ฐ๋ฉฐ ํฐ ๊ฐ์ ๊ฐ์ง ์๋ก ๋งค์ฐ ๋์ ์ฑ๋ฅ์ผ๋ก ์๋ ดํ๋ ๋ชจ์ต**์ ๋ณด์ด๊ณ  ์์ต๋๋ค. 
    - ๋ฐ๋ฉด **<span style="color:maroon">Bagging์ ๊ฒฝ์ฐ ์ด๋ฐ์๋ ์ฑ๋ฅ์ด ์ค๋ฅด๋ค๊ฐ, ์ดํ์๋ ์ฑ๋ฅ์ด ์ฝ๊ฐ ๋จ์ด์ง๊ฑฐ๋ ๋ค์ญ๋ ์ญํ ๊ฒฝํฅ**์ด ์์ต๋๋ค.
- ๋ค์ ๋งํด, **<span style="color:maroon">k ๊ฐ์ด ์ปค์ง๋ฉด Bagging์ ํจ๊ณผ๊ฐ ํฌ๊ฒ ๋์ค์ง ์๋ ๊ฒ**์ด์ฃ . ์ด๋ ํนํ AUROC์์ ํ์คํ ๊ฒฝํฅ์ ํ์ธํ  ์ ์์ต๋๋ค.
- **<span style="color:maroon">๊ฒฐ๋ก ์ ์ผ๋ก, ์ด ๋ฐ์ดํฐ์์๋ k=6~7 ์ ๋๋ก ์ก์ ๋ Bagging์ ํจ๊ณผ๋ฅผ ๋ณผ ์ ์๋ค๊ณ  ๊ฒฐ๋ก ** ์ง์ ์ ์์ต๋๋ค.

<br/>

# <span style="color:darkblue">Bagging 2: <span style="color:green">Random Forest ๐ฒ๐ณ๐ด

<p align="center">
    <img src="Img/RF.png" width="500"/>
</p>

<p align="center">
    <em>Random Forest: General Framework </em>
</p>
<p align="center">
    <em> Image source: https://ai-pool.com/a/s/random-forests-understanding </em>
</p>

Bagging ๋ฐฉ๋ฒ๋ก ์ ๊ธฐ๋ฐ์ผ๋ก ํ๋ ๋ํ์ ์ธ ์๊ณ ๋ฆฌ์ฆ์ Random Forest์๋๋ค. Decision Tree๋ฅผ ์ฌ๋ฌ ๊ฐ ๋ชจ์ ๋์ผ๋ฉด ์ฒ์ด ๋๋๋ฐ, ์ด ์ฒ์ ๊ตฌ์ฑํ๋ ๋ฐฉ๋ฒ์ Random์ผ๋ก ํ๋ค๊ณ  ํ์ฌ "Random Forest"๋ก ๋ถ๋ฆฌ๋ ๋ชจ๋ธ์ด์ฃ .   

๋ณด๋ค ๊ตฌ์ฒด์ ์ผ๋ก๋ **์ฌ๋ฌ ๊ฐ์ Decision Tree๋ฅผ ์์ฑํ ๋ค, ๊ฐ ๊ฐ๋ณ Tree์ ์์ธก ๊ฐ๋ค ์ค ๊ฐ์ฅ ๋ง์ ์ ํ์ ๋ฐ์ ๋ณ์๋ค๋ก ์์ธก์ ์งํํ๋ ๋ฐฉ์์ผ๋ก ๋์**ํฉ๋๋ค. Decision Tree์ ์ค์ฌ ๊ทนํ ์ ๋ฆฌ ๋ฒ์ ์ด๋ผ ํ  ์ ์์ฃ .   

์ด๋ฌํ ๋ฐฉ์์ ์ฅ์ ์, **์์ธก ๊ฐ์ ๋ํ Variance๊ฐ ๋๋ค ํ๋๋ผ๋, ์ด๋ฅผ ํ๊ท ๋ด์ ๋ถ์ฐ์ ์ค์ผ ์ ์๋ค๋ ๊ฒ**์๋๋ค.   

๊ทธ๋ฆฌ๊ณ  ๊ฐ Decision Tree๋ง๋ค ๋๋ฆฝ๋ณ์์ ์ฌ์ฉ ๊ฐ์๋ฅผ ์ ํํ๋๋ฐ, ์ด๋ Bagging ๊ธฐ๋ฒ์ ์ฌ์ฉํ๋ ๊ฒ์ด ํน์ง์ ์ด์ฃ . Random Forest๋ ๊ธฐ๋ณธ์ ์ผ๋ก๋ Bagging์ ๋ฐฉ์์ ๋ฐ๋ฅด๊ธฐ ๋๋ฌธ์, **<span style="color:green">๊ฐ Decision Tree ๋ง๋ค ์ฌ์ฉ๋๋ ๋ฐ์ดํฐ์(Bootstrap)์ ๋ค๋ฅด์ง๋ง, Bagging์ฒ๋ผ ๋ชจ๋  ๋ณ์๋ฅผ ์ฌ์ฉํ๋ ๋์ ์ Tree ๋ณ๋ก ํ์ฉํ๋ ๋๋ฆฝ๋ณ์๋ฅผ ๋ค๋ฅด๊ฒ ํ๋ ๊ธฐ๋ฒ**์๋๋ค.   

์ด๋ ํ์ฉํ๋ ๋๋ฆฝ๋ณ์์ ์๋ ์๋ ๋ณ์์ ์ $D$๋ณด๋ค ์ ์ ์์ ๋ณ์๋ฅผ ์ฌ์ฉํ๊ณ , ๋ณดํต $\sqrt D$๊ฐ๋ฅผ ์ฌ์ฉํฉ๋๋ค.

`Scikit-Learn`์์๋ Classification Task์ ์์ด Random Forest๋ฅผ ๊ฐํธํ๊ฒ ์ด์ฉํ  ์ ์๋๋ก, `RandomForestClassifier`๋ฅผ ์ ๊ณตํ๊ณ  ์์ต๋๋ค.
```python
from sklearn.ensemble import RandomForestClassifier
```

<br/>

## <span style="background-color:#fff5b1"> [์คํ 3] Bagging with Decision Tree vs. RandomForest
๊ทธ๋ ๋ค๋ฉด ์ฌ๊ธฐ์ ์ง๋ฌธ์ด ํ๋ ์๊น๋๋ค. **<span style="color:darkblue">Decision Tree๋ฅผ Base Learner๋ก ์ฌ์ฉํ๋ Bagging ๋ฐฉ์</span>๊ณผ <span style="color:green">Random Forest</span> ๊ฐ์ ์ฑ๋ฅ ์ฐจ์ด๋ ์ผ๋ง๋ ๋ ๊น์?**   

๋ชจ๋  ๋ณ์๋ฅผ ์ฌ์ฉํ๋ฉด ๋ ์ ๋ณด๋์ด ๋ง์ผ๋, Bagging์ด ๋ ๋์ ์ฑ๋ฅ์ ๋ผ ์ ์์๊น์? ์๋๋ฉด ๋ค์ํ ์๋ ฅ ๋ณ์ ์กฐํฉ์ ๋ํ ์ฑ๋ฅ์ ํ์ธํ  ์ ์์ผ๋, Random Forest์ ์ฑ๋ฅ์ด ๋ ์ข๊ฒ ๋์ฌ๊น์? ์ด๋ฅผ ์คํ์ ์ผ๋ก ํ์ธํด ๋ณด๊ฒ ์ต๋๋ค.
```python
# Random Forest
rf_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=15, n_jobs=-1, random_state=2022)

# Bagging with Decision Tree
bag_clf_rf = BaggingClassifier(DecisionTreeClassifier(splitter="random", max_leaf_nodes=15, 
            random_state=2022), n_estimators=500, bootstrap=True, n_jobs=-1, random_state=2022)
```

- Bagging๊ณผ RandomForest ๋ชจ๋ Tree์ ๊ฐ์๋ 500๊ฐ, max_leaf_nodes 15๊ฐ๋ก ๋์ผํ ์กฐ๊ฑด์ ์ฃผ์์ต๋๋ค. ๋ค๋ง ๋ค๋ฅธ ๊ฒ์ ์ฌ์ฉํ๋ ์๋ ฅ ๋ณ์์ ๊ฐ์๊ฐ ๋ค๋ฅด๊ฒ ์ฃ .   

```python
# Evaluate Random Forest
acc_RF, acc_RF = evaluate_model(model=rf_clf, X=X, y=y)

# Evaluate Bagging with Decision Tree
acc_bag_DT, auroc_bag_DT = evaluate_model(model=bag_clf_rf, X=X, y=y)
```

### Results
- Random Forest: 0.88(0.03), 0.96(0.02)
- Bagging: 0.87(0.03), 0.95(0.02)

### <span style="background-color:#fff5b1"> [์คํ 3] ๊ฒฐ๊ณผ ํด์
- Decision Tree๋ฅผ ๊ธฐ๋ฐ์ผ๋ก ํ๋ Bagging๋ณด๋ค๋, **<span style="color:maroon">Random Forest์ ์ฑ๋ฅ์ด ๋ ์ฐ์ํ ๊ฒ์ ์คํ์ ์ผ๋ก ํ์ธ** ํ์์ต๋๋ค. ๋์ ์ฐจ์ด๋ Bootstrap ๋ง๋ค ํ์ฉํ๋ ์๋ ฅ ๋ณ์์ ๊ฐ์์๋๋ฐ, **<span style="color:maroon">Random Forest๋ ์๋ก ๋ค๋ฅธ ์๋ ฅ ๋ณ์ ์กฐํฉ์ ํ์ฉํ๋ค๋ ํน์ง**์ ๊ฐ์ง๊ณ  ์์์ฃ .
- ์ด๋ก์จ ์ ์ ์๋ ๊ฒ์, Ensembel Learning์ ์์ด **<span style="color:maroon">๊ฐ๋ณ ๋ชจ๋ธ์ ๋ค์์ฑ ํ๋ณด**์๋๋ค. Ensemble Learnig์ ์์ด ๊ฐ์ฅ ์ค์ํ ํต์ฌ ์์ด๋์ด๋, ๊ฐ๋ณ ๋ชจ๋ธ์ "๋ค์์ฑ"์ ์ด๋ป๊ฒ ํ๋ณดํ  ๊ฒ์ธ๊ฐ?์ ๊ธฐ๋ฐํ๊ธฐ ๋๋ฌธ์ด์ฃ . ๋์ผํ ๋ชจ๋ธ์ ์ฌ๋ฌ ๊ฐ ์ทจํฉํด๋ดค์ ํฐ ์ฑ๋ฅ ํฅ์์ด ์์ ํ๋๊น์. ์ฌ๊ธฐ์ **<span style="color:maroon">"๋ค์ํ ๋ชจ๋ธ"์ด๋ผ ํจ์, ์๊ณ ๋ฆฌ์ฆ์ด ๋ค๋ฅด๊ฑฐ๋, ๋์ผ ์๊ณ ๋ฆฌ์ฆ์ด์ด๋ ๋ฐ์ดํฐ์ ๊ตฌ์ฑ์ด๋ ํ์ดํผํ๋ผ๋ฏธํฐ๋ฅผ ๋ฌ๋ฆฌ ํ์ฌ ํ์ตํ ์ํฉ์ ์๋ฏธ**ํฉ๋๋ค. (๋ฐ์ดํฐ ๋ด์ ๋ธ์ด์ฆ๊ฐ ์๋ก ๋ค๋ฅด๊ณ , ํ์ดํผํ๋ผ๋ฏธํฐ๊ฐ ์๊ณ ๋ฆฌ์ฆ์ ๊ตฌ์กฐ์ ์ํฅ์ ์ฃผ๊ธฐ ๋๋ฌธ์ด์ฃ .) ์ด๋ฌํ ์ํฉ์์ ๊ฐ๋ณ ๋ชจ๋ธ์ ์๋ก ์ ์ ํ๊ฒ ๋ฌ๋ผ์ผ ์์๋ธ์ ํจ๊ณผ๋ฅผ ๋ณผ ์ ์์ต๋๋ค.
- ๋ฐ๋ผ์ Ensemble Learning์ ํจ๊ณผ๋ฅผ ๋ณด๊ธฐ ์ํด์๋ **<span style="color:maroon">๊ฐ๋ณ์ ์ผ๋ก๋ ์ด๋ ์ ๋์ ์ข์ ์ฑ๋ฅ(Random Model๋ณด๋ค๋ ์ข์ ์ฑ๋ฅ)์ ๊ฐ์ง๊ณ , ์์๋ธ ๋ด์์ ๊ฐ๊ฐ์ ๋ชจ๋ธ์ด ์๋ก ๋ค์ํ ํํ๋ฅผ ๋ํ๋ด๋ ๊ฒ์ด ๊ฐ์ฅ ์ด์์ **์๋๋ค.
- ๊ทธ๋ฐ๋ฐ **<span style="color:maroon">Bagging์ ๋ฐ์ดํฐ๋ ๋ค๋ฅด์ง๋ง ์๋ ฅ ๋ณ์๊ฐ ๋ชจ๋ ๋์ผํ ๋ฐ๋ฉด, Random Forest๋ ์๋ ฅ ๋ณ์๊ฐ ๋ชจ๋ธ ๋ณ๋ก ๋ชจ๋ ๋ฌ๋ผ ๋ค์์ฑ ์ธก๋ฉด์์ ์ฐ์ํ๋ค๊ณ  ํ  ์ ์์ฃ . ์ด๊ฒ์ด ๋ฐ๋ก Random Forest๊ฐ ๋จ์ Decision Tree์ Bagging๋ณด๋ค ๋ ์ข์ ์ฑ๋ฅ์ ๋ด๋ ์ด์ **๋ผ ํ  ์ ์์ต๋๋ค.

<br/>

## <span style="background-color:#fff5b1"> [์คํ 4] Random Forest ์ฌ์ฉ ๋ณ์ ๊ฐ์ ๋ณ ์ฑ๋ฅ ์ฐจ์ด๊ฐ ์์๊น?
๊ทธ๋ฌ๋ ์ ์คํ์์ Bagging๊ณผ Random Forest ๊ฐ ์ฑ๋ฅ ์ฐจ์ด๊ฐ ํฌ์ง๋ ์์์ต๋๋ค. ๊ทธ๋ ๋ค๋ฉด ์ด๋ฐ ์ง๋ฌธ์ด ์๊ธธ ์๋ ์์ต๋๋ค. **์ ๋ง Random Forest์ ์ฌ์ฉ ๋ณ์ ๊ฐ์๋ $\sqrt D$๊ฐ์ด๋ฉด ๋ ๊น์?** ์ด๊ฒ์ด ์ต์ ์ ์ฑ๋ฅ์ ๋ด๋ ๊ฒ์ผ๊น์?   
์ด๋ฐ ๋ฌผ์์ ๊ฐ์ง๊ณ  [์คํ 4]์์๋ Random Forest์์ Bootstrap๋ง๋ค ์ฌ์ฉ๋๋ ์๋ ฅ๋ณ์์ ์๋ฅผ 1๋ถํฐ ์๋ ฅ ๋ณ์์ ๊ฐ์๋งํผ ๋๋ ค๊ฐ๋ฉฐ ์ฑ๋ฅ์ ๋น๊ตํด๋ณด์์ต๋๋ค.

```python
def get_RF():
    random_forest = dict()
    # Evaluate 'max_features' = from 1 to 20
    for i in range(1, 20+1):
        # Define Random Forest
        random_forest[str(i)] = RandomForestClassifier(n_estimators=500, n_jobs=-1, max_features=i, random_state=2022)
    return random_forest
```
```python
rf_clf_dict = get_RF()
```
```python
def print_results_per_max_features(models:dict, X, y):
    acc_score, auroc_score, n_feature_list = [], [], []
    for k, model in models.items():
        # Evaluate Random Forest
        acc, auroc = evaluate_model(model, X, y)
        # Store the Results
        acc_score.append(acc)
        auroc_score.append(auroc)
        n_feature_list.append(k)
        
        # Print the performance along the way
        print('n_feature=%s, Accuracy: %.3f (%.3f)' % (k, np.mean(acc), np.std(acc)))
        print('n_feature=%s, AUROC: %.3f (%.3f)' % (k, np.mean(auroc), np.std(auroc)))
        
    return acc_score, auroc_score, n_feature_list
```
```python
rf_acc_list, rf_auroc_list, rf_n_feature_list = print_results_per_max_features(rf_clf_dict, X, y)
```
```
Accuracy: 0.93 (0.03)
AUROC: 0.98 (0.01)
n_feature=1, Accuracy: 0.932 (0.027)
n_feature=1, AUROC: 0.983 (0.008)
Accuracy: 0.94 (0.02)
AUROC: 0.98 (0.01)
n_feature=2, Accuracy: 0.936 (0.023)
n_feature=2, AUROC: 0.984 (0.008)
...
Accuracy: 0.90 (0.03)
AUROC: 0.97 (0.01)
n_feature=20, Accuracy: 0.905 (0.032)
n_feature=20, AUROC: 0.970 (0.013)
```
```python
Best_Acc_idx = np.where([np.mean(acc) for acc in rf_acc_list] == np.max([np.mean(acc) for acc in rf_acc_list]))[0][0]+1
print('Best "max_features" is', Best_Acc_idx,
      'in terms of Accuracy:', '%.3f' % (np.mean(rf_acc_list[Best_Acc_idx-1])))
```
```
Best "max_features" is 2 in terms of Accuracy: 0.936
```
```python
Best_AUROC_idx = np.where([np.mean(auroc) for auroc in rf_auroc_list] == np.max([np.mean(auroc) for auroc in rf_auroc_list]))[0][0]+1
print('Best "max_features" is', Best_AUROC_idx,
      'in terms of AUROC:', '%.3f' % (np.mean(rf_auroc_list[Best_AUROC_idx-1])))
```
```
Best "max_features" is 3 in terms of AUROC: 0.985
```

### Result
```python
plt.figure(figsize=(10, 5))
plt.title("Random Forest num of features vs. Classification accuracy")
plt.boxplot(x=rf_acc_list, labels=rf_n_feature_list, showmeans=True);
```
<p align="center">
    <img src="Img/RF_acc_features.png" width="500"/>
</p>

```python
plt.figure(figsize=(10, 5))
plt.title("Random Forest num of features vs. Classification AUROC")
plt.boxplot(x=rf_auroc_list, labels=rf_n_feature_list, showmeans=True);
```

<p align="center">
    <img src="Img/RF_auroc_features.png" width="500"/>
</p>

### <span style="background-color:#fff5b1"> [์คํ 4] ๊ฒฐ๊ณผ ํด์
- ์ด 20๊ฐ์ ๋ณ์ ์ค, Random Forest ๋ด Subset์์ ๋ช ๊ฐ์ ๋ณ์๋ฅผ ์๋ ฅ ๋ฐ์ ๊ฒ์ธ์ง(`max_features`)์ ๋ฐ๋ฅธ ์ฑ๋ฅ ๋ณํ๋ฅผ Boxplot์ผ๋ก ํ์ธํด ๋ณด์์ต๋๋ค.
- ์ ์ฒด์ ์ธ ๊ฒฝํฅ์ ํ์ธํด๋ณด๋, **<span style="color:maroon">์ฌ์ฉํ๋ ๋ณ์์ ์๊ฐ ๋ง์์ง ์๋ก ์คํ๋ ค ์ฑ๋ฅ์ด ํ๋ฝํ๋ ์ผ๋ฐ์ ์ธ ๊ฒฝํฅ์ Accuracy์ AUROC ๊ด์ ์์ ๋ชจ๋ ํ์ธ**ํ  ์ ์์์ต๋๋ค.
- ํนํ Acuuracy ๊ธฐ์ค์์๋ ์๋ ฅ ๋ณ์๋ฅผ 2๊ฐ๋ง ์ฌ์ฉํ  ๋, ๊ทธ๋ฆฌ๊ณ  AUROC ๊ธฐ์ค์์๋ ์๋ ฅ ๋ณ์๋ฅผ ์ค์ง 3๊ฐ๋ง ์ฌ์ฉํ  ๋ ๊ฐ์ฅ ์ข์ ์ฑ๋ฅ์ ๋ณด์์ต๋๋ค. ๋ชจ๋ $\sqrt D$ ($D=20 in this case$) ๋ณด๋ค๋ ์์ ๊ฐ์๋๋ค.   
- ์ด๋ ๊ฒ **<span style="color:maroon">์ ์ ๋ณ์๋ง์ ํ์ฉํด๋ ์ข์ ์ฑ๋ฅ์ด ๋์ค๋ ์ด์ ๋, [Dimensionality Reduction Tutorial](https://github.com/Im-JihyunKim/BusinessAnalytics_Topic1)๋ ๋ค๋ฃจ์๋ฏ์ด, ํ์ฉํ๋ ๋ณ์ ์ฐจ์์ด ๋์ด๋  ์๋ก ์ฐจ์์ ์ ์ฃผ์ ๋น ์ ธ๋ค๊ธฐ ์ฝ๊ณ , ๋ํ ๊ฐ๋ณ ๋ชจ๋ธ์ ๋ค์์ฑ์ ํ๋ณดํ๊ธฐ ์ฉ์ด** ๋ค๋ ์ฅ์  ๋๋ฌธ์๋๋ค.

<br/>

-------

<br/>

# <span style="color:purple">Boosting 1: <span style="color:darkviolet">Gradient Boosting
<p align="center">
    <img src="Img/gradientboosting.png" width="500"/>
</p>

<p align="center">
    <em>Gradient Boosting: General Framework   </em>
</p>
<p align="center">
    <em> Image source: https://www.geeksforgeeks.org/ml-gradient-boosting/ </em>
</p>

Gradient Boosting์ ๋ํ์ ์ธ Boosting ๊ณ์ด ์๊ณ ๋ฆฌ์ฆ์ผ๋ก์, XGBoost, LightGBM, CatBoost์ ๊ทผ๊ฐ์ด ๋๋ ์๊ณ ๋ฆฌ์ฆ์๋๋ค. **<span style="color:darkviolet">ํ์ต์ ์  ๋จ๊ณ์์ ๋ชจ๋ธ ๋ณ ์์ฌ ์ค์ฐจ(residual error)๋ฅผ ๊ณ์ฐํ๊ณ , ์ด ์ด ์ค์ฐจ๋ฅผ ๋ฏธ๋ถํ gradient๋ฅผ ํตํด ๋ชจ๋ธ์ ๋ณด์ํ๋ ๋ฐฉ์**์ ์ทจํฉ๋๋ค. ๊ทธ๋ ๊ธฐ ๋๋ฌธ์ "Gradient" Boosting์ด๋ผ๋ ์ด๋ฆ์ด ๋ถ์์ต๋๋ค.   

Gradient Boosting์์ ์ฌ์ฉ๋๋ ๊ฐ์ฅ ํต์ฌ์ ์ธ ๋ฐฉ๋ฒ์ **<span style="color:darkviolet">Gradient Descent, ์ฆ ๊ฒฝ์ฌ ํ๊ฐ๋ฒ**์๋๋ค. Gradient Descent๋ **<span style="color:darkviolet">Loss function์ ์ ์ํ๊ณ , ์ด ๋ฏธ๋ถ๊ฐ์ด ์ต์ํ๋๋ ๋ฐฉํฅ์ ์ฐพ์๋๊ฐ๋ ๋ฐฉ์**์๋๋ค. ๋ง์ผ Loss Function์ Squared Error๋ก ์ ์ํ๋ค๋ฉด, ์๋์ ๊ฐ์ ์์ผ๋ก Loss์ Loss์ ๋ฏธ๋ถ ๊ฐ์ ํํํ  ์ ์๊ฒ ์ฃ .   

$$L(y_i, F(x_i)) = \frac{1}{2}(y_i - F(x_i))^2$$
$$\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} = y_i - F(x_i)$$

์ด๋ ํธ๋ฏธ๋ถ์ ํตํด ์ป์ gradient๊ฐ ๊ฒฐ๊ตญ $f(x)$๊ฐ Loss๋ฅผ ์ค์ด๊ธฐ ์ํด ๊ฐ์ผํ๋ ๋ฐฉํฅ์ธ๋ฐ, ์ด๊ฒ์ด ๊ฒฐ๊ตญ ์์ฌ ์ค์ฐจ(residual error)์ ๊ฐ์ต๋๋ค. ์ด๋ ํ๊ท ๋ชจํ์ ์์ฐจ๋ Squared Loss Function์ Negative gradient $y_i-F(x_i)=-\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}$๋ฅผ ์ฌ์ฉํฉ๋๋ค.   

๊ทธ๋ ๋ค๋ฉด Gradient Boosting์์ ์ฌ์ฉ๋๋ ์์์ ๋ฌด์์ผ๊น์?   

$$train \ set: \begin{Bmatrix}
(x_i, y_i)
\end{Bmatrix}^N_{i=1} \quad loss \ funtion: L(y, F(x))$$
$$F_0(x) = arg \ \underset{\gamma }{min}\sum_{i=1}^{N}L(y_i, \gamma )$$

n๊ฐ์ ํ์ต ๋ฐ์ดํฐ๊ฐ ์์ ๋, Gradient Boosting์์๋ ์ด๊ธฐ ๊ฐ์ผ๋ก ์์ ํจ์๋ฅผ ํ์ฉํฉ๋๋ค.  ๊ทธ๋ฆฌ๊ณ  ์๋์ ๊ฐ์ด pseudo-residual, ์ฆ gradient๋ฅผ ๊ณ์ฐํฉ๋๋ค. ์ด๋ฅผ python์ผ๋ก ๊ตฌํํ๋ฉด ์๋์ ๊ฐ์ต๋๋ค.   

๊ฐ๋จํ ์์๋ฅผ ์ํด MSE Loss๋ฅผ ์ต์ํ ํ๋ Regression Task๋ฅผ ํผ๋ค๊ณ  ๊ฐ์ ํ๊ฒ ์ต๋๋ค.
```python
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
```
```python
# Define dataset
X, y = make_regression(n_samples=100, n_features=1, random_state=2022)
```
์์์ Regression์ ์ํ ๊ฐ์ ๋ฐ์ดํฐ์์ ๋ง๋ค๊ณ , ์ด๋ฅผ Single Decision Tree์ ํ์ต์ํต๋๋ค. max_depth๋ฅผ 2๋ก ๋์ด Bias๊ฐ ๋์ Weak Learner๋ฅผ ์ฌ์ฉํฉ๋๋ค.

```python
F0 = np.mean(y)
print(F0)
```
```
-0.2068375978691931
```
์ฒซ ๋ฒ์งธ Update๋ ๋จผ์  $y$ ๊ฐ์ ํ๊ท ์ผ๋ก ๋ชจํ์ ์ด๊ธฐํํฉ๋๋ค.
$$F_0(x) = -0.207$$

```python
# 1st residual error
r1 = y - F0

# First Single Decision Tree with 'max_depth' = 2
tree_1 = DecisionTreeRegressor(max_depth=2, random_state=2022, criterion='mse')
r1_fit = tree_1.fit(X, r1)
gamma1 = r1_fit.predict(X)
```
์ดํ ์ฒซ ๋ฒ์งธ residual error $r_1 = y-F_0(x)$๋ฅผ ๊ตฌํ๊ณ , ์ด ์์ฐจ์ max_depth=2์ธ Single Decision Tree๋ฅผ ํ์ต์์ผ ์์ฐจ์ ์์ธก ๊ฐ์ธ $\gamma$๋ฅผ ๊ตฌํฉ๋๋ค.
```python
print(f"Prediction of residual: {np.unique(gamma1)}")
```
```
Prediction of residual: [-42.66613158 -12.51579569  10.53423901  30.43568584]
```
Residual์ด ์ผ์ข์ pseudo target value๊ฐ ๋๋ ๊ฒ์ด์ฃ .
```python
lr = 0.1
F1 = F0 + lr * gamma1 
```
์ด์  ์์ธก ๊ฐ์ ์๋ฐ์ดํธ ํด์ค๋๋ค. ์ฌ๊ธฐ์ `lr`์ learning rate, ์ฆ gradient ๊ณ์ฐ์ ์์ด ํ์ต๋ฅ ์ ์๋ฏธํฉ๋๋ค. `gamma1`์ ์ฒซ ๋ฒ์งธ ์์ฐจ๋ฅผ ์์ธกํ ๊ฐ, `F1`์ ์๋ก ์์ธก๋ $y$ ๊ฐ์ ์๋ฏธํฉ๋๋ค.
```python
print(f"1st Prediction: {F1}")
```
```
1st Prediction: [-1.45841717 -4.47345076 -4.47345076  0.8465863  -1.45841717  0.8465863
  2.83673099  2.83673099 -1.45841717 -1.45841717  0.8465863   0.8465863
 -1.45841717 -1.45841717  0.8465863   0.8465863   2.83673099  0.8465863
 -1.45841717  2.83673099  2.83673099  2.83673099  0.8465863   0.8465863
  2.83673099 -1.45841717  0.8465863  -4.47345076  2.83673099 -4.47345076
...
 -1.45841717 -1.45841717  0.8465863   0.8465863   2.83673099 -4.47345076
 -4.47345076  2.83673099  0.8465863   0.8465863  -1.45841717 -1.45841717
 -1.45841717  2.83673099  2.83673099  2.83673099]
```
์์ธก ๊ฐ์ ๋ณด๋ฉด ๊ฑฐ์ ๋์ผํ ๊ฐ์ผ๋ก ์์ธก์ ํ๋ฉฐ, ๊ทธ ์ฑ๋ฅ์ด ํํธ ์์์ ์ ์ ์์ต๋๋ค. Single Weak Learner๋ก๋ ํ ๋ฒ๋ง์ผ๋ก๋ ์ข์ ์์ธก ์๋ฅ์ ๋ด์ง ๋ชปํ๋ ๊ฒ์ด์ฃ . ํ์ง๋ง $T$๋ฒ ๋งํผ ํ์ต์ ๋ฐ๋ณตํ๋ค๋ฉด ์ด๋ป๊ฒ ๋ ๊น์? ์ ๊ณผ์ ์ $T$๋ฒ ๋ฐ๋ณตํ  ์ ์๋ ํจ์๋ ์๋์ ๊ฐ์ต๋๋ค.
```python
def GBM_Regression(X, y, T: int, lr):
    F_t = np.mean(y)
    tree = DecisionTreeRegressor(max_depth=2, random_state=2022)  # Weak Learner
    
    for t in range(1, T+1): # t=100
        print("Current update: ", str(t))
        F_before_t = F_t
        residual = y - F_before_t
        residual_fit = tree.fit(X, residual)
        gamma = residual_fit.predict(X)
        print(f"Prediction of Residual: {np.round(np.unique(gamma), 2)}")
        
        F_t = F_before_t + lr * gamma
        
        print(f"Prediction: {np.round(F_t, 2)}")
```
```python
GBM_Regression(X, y, T=100, lr=1e-2)
```
```
Current update:  1
Prediction of Residual: [-42.67 -12.52  10.53  30.44]
Prediction: [-0.33 -0.63 -0.63 -0.1  -0.33 -0.1   0.1   0.1  -0.33 -0.33 -0.1  -0.1
 -0.33 -0.33 -0.1  -0.1   0.1  -0.1  -0.33  0.1   0.1   0.1  -0.1  -0.1
  0.1  -0.33 -0.1  -0.63  0.1  -0.63 -0.33 -0.1  -0.33 -0.33 -0.33  0.1
 -0.1  -0.33 -0.63 -0.1  -0.33 -0.33 -0.33 -0.1  -0.1  -0.33 -0.33 -0.1
 -0.1   0.1  -0.63 -0.1  -0.33  0.1  -0.33 -0.1   0.1  -0.1  -0.63 -0.33
 -0.33 -0.33 -0.63 -0.63  0.1  -0.33 -0.1  -0.33 -0.1   0.1  -0.33  0.1
 -0.1  -0.1  -0.33 -0.33  0.1  -0.33 -0.63 -0.33 -0.33  0.1  -0.33 -0.1
 -0.33 -0.33 -0.1  -0.1   0.1  -0.63 -0.63  0.1  -0.1  -0.1  -0.33 -0.33
 -0.33  0.1   0.1   0.1 ]
Current update:  2
Prediction of Residual: [-43.55 -13.47   8.68  28.84]
Prediction: [-0.47 -1.07 -1.07 -0.01 -0.47 -0.01  0.39  0.39 -0.47 -0.47 -0.01 -0.01
 -0.47 -0.47 -0.01 -0.01  0.39 -0.01 -0.47  0.39  0.39  0.39 -0.01 -0.01
  0.39 -0.47 -0.01 -1.07  0.39 -1.07 -0.47 -0.01 -0.47 -0.47 -0.47  0.39
 -0.01 -0.47 -1.07  0.19 -0.47 -0.47 -0.25 -0.01 -0.01 -0.47 -0.47 -0.01
 -0.01  0.39 -0.77  0.19 -0.47  0.39 -0.47 -0.01  0.39 -0.01 -1.07 -0.47
 -0.47 -0.47 -1.07 -1.07  0.39 -0.47 -0.01 -0.47 -0.01  0.39 -0.47  0.39
 -0.01 -0.01 -0.47 -0.47  0.39 -0.47 -1.07 -0.47 -0.25  0.39 -0.47 -0.01
 -0.47 -0.47 -0.01 -0.01  0.39 -1.07 -1.07  0.39  0.19 -0.01 -0.47 -0.47
 -0.47  0.39  0.39  0.39]

 ...

Current update:  100
Prediction of Residual: [-18.42  -4.15   6.89  25.6 ]
Prediction: [-2.060e+00 -2.353e+01 -2.353e+01  6.080e+00 -1.040e+01  8.420e+00
  2.618e+01  1.555e+01 -9.810e+00 -1.170e+01 -2.000e-02  6.080e+00
 -7.800e+00 -1.040e+01  3.670e+00  6.080e+00  1.526e+01  4.160e+00
 -1.442e+01  1.479e+01  2.618e+01  1.622e+01  6.080e+00  1.029e+01
  1.622e+01 -1.442e+01  7.360e+00 -2.353e+01  1.555e+01 -2.336e+01
 -7.020e+00  2.880e+00 -1.040e+01 -8.390e+00 -1.544e+01  1.375e+01
  6.080e+00 -1.930e+00 -2.452e+01  1.244e+01 -2.060e+00 -1.626e+01
 -6.500e-01  3.700e-01  9.830e+00 -4.140e+00 -1.301e+01  1.960e+00
  6.080e+00  1.622e+01 -1.740e+01  1.244e+01 -3.790e+00  1.479e+01
 -4.840e+00  6.080e+00  1.526e+01  9.830e+00 -3.320e+01 -3.790e+00
 -1.205e+01 -6.080e+00 -2.883e+01 -2.374e+01  1.555e+01 -5.360e+00
  8.580e+00 -2.480e+00  9.370e+00  1.526e+01 -2.060e+00  2.618e+01
  6.080e+00  8.580e+00 -2.200e+00 -1.442e+01  1.622e+01 -1.040e+01
 -3.320e+01 -1.110e+01 -8.300e-01  1.555e+01 -1.780e+00  6.080e+00
 -1.170e+01 -1.442e+01  9.830e+00  6.080e+00  1.685e+01 -2.353e+01
 -2.452e+01  2.618e+01  1.244e+01  6.080e+00 -8.840e+00 -1.040e+01
 -1.222e+01  1.339e+01  1.555e+01  1.526e+01]
```
์ด๋ฐ์ Update ๊ฐ์ ์์ฐจ๋ ๋งค์ฐ ํฌ๊ณ , ์์ธก ๊ฐ๋ ์ผ์ ํ ์์ค์ผ๋ก ๋ฐ์ด๋ฒ๋ฆฌ์ง๋ง, 100๋ฒ ๊ฐ๋์ Update ํ์๋ ์์ฐจ ๊ฐ์ด ์๋์ ์ผ๋ก ๋ฎ์์ก์ผ๋ฉฐ, data point ํ๋ํ๋์ ๋ํ ์ค์  prediction ๊ฐ์ ๋ด๋ฑ๋ ๊ฒฝํฅ์ด ๊ฐํด์ง๋๋ค.

$$g_{im} = \begin{bmatrix}
\frac {\partial L(y_i, f(x_i))}{\partial f(x_i)}
\end{bmatrix}_{f(x_i) = f_{m-1}(x_i)}$$
<br/>

์์์ ๋ณธ ๊ณผ์ ๊ณผ ๊ฐ์ด, **<span style="color:darkviolet">Gradient Boosting์ ํ์ต ๋ฐ์ดํฐ์ y ๋์  gradient๋ฅผ ์ ์ฉํ๊ณ (target์ pseudo residual์ ์ ์ฉํ๋ ๊ฒ์ด์ฃ ), Loss function์ ๋ฃ์ผ๋ฉด์ ๊ณ์ํด์ ์์ฐจ๋ฅผ ์ค์ด๋ ๋ฐฉ์**์ ํํฉ๋๋ค.   

<br/>

$$h_t(x): base \ model(tree) \quad train \ set: \begin{Bmatrix}
(x_i, g_{im}) \\
\end{Bmatrix}^N_{i=1} \\ 
F_t(x) = F_{t-1}(x) + \alpha h_t(x)$$

๋ฐ๋ผ์ **<span style="color:darkviolet">์ฒ์ $F_0(x)$๋ ์์ํจ์์์ง๋ง (In this case, $F_0 = mean\ of\ y\ values$), ํ์ฌ ์์  t์ ๋ํ ๋ชจ๋ธ $h_t(x)$๊ฐ ๋ค์ด๊ฐ๋ฉฐ gradient๋ฅผ ๊ณ ๋ คํ ํ์ต์ด ๊ฐ๋ฅ**์ผ ๋์์ฃ . ์ฐธ๊ณ ๋ก $\alpha$๋ Learning rate (lr) ์๋๋ค. ์์ ์ฒ๋ผ ์ง์  ๋ฃ๊ฑฐ๋ ์ต์ ํ ์์ ๋ฃ์ด ์ฌ์ฉํ๊ธฐ๋ ํฉ๋๋ค.   

์ด ๊ณผ์ ์ python์ ํตํด ์๊ฐํํ์๋ฉด ์๋์ ๊ฐ์ต๋๋ค.
```python
# First Update
F0 = np.mean(y)
r1 = y - F0

# Second Update
tree_1 = DecisionTreeRegressor(max_depth=2, random_state=2022, criterion='mse')
tree_1.fit(X, r1)

# Third Update
y2 = y - tree_1.predict(X)  # residual errors

tree_2 = DecisionTreeRegressor(max_depth=2, random_state=2022, criterion='mse')
tree_2.fit(X, y2)

# Final Update
y3 = y2 - tree_2.predict(X)

tree_3 = DecisionTreeRegressor(max_depth=2, random_state=2022)
tree_3.fit(X, y3)

# Prediction
y_pred = sum(tree.predict(X_test) for tree in (tree_1, tree_2, tree_3))
```
```python
def plot_predictions(regressors, X, y, axes, label=None, style="r-", data_style="b.", data_label=None):
    x1 = np.linspace(axes[0], axes[1], 500)
    y_pred = sum(regressor.predict(x1.reshape(-1, 1)) for regressor in regressors)
    plt.plot(X[:, 0], y, data_style, label=data_label)
    plt.plot(x1, y_pred, style, linewidth=2, label=label)
    if label or data_label:
        plt.legend(loc="upper center", fontsize=16)
    plt.axis(axes)
```
```python
plt.figure(figsize=(11,11))

axes = [X.min()-1, X.max()+1, y.min()-1, y.max()+1]

plt.subplot(321)
plot_predictions([tree_1], X, y, axes=axes, label="$h_1(x_1)$", style="g-", data_label="Train Data")
plt.ylabel("$y$", fontsize=12, rotation=0)
plt.title("Residual Error & Prediction of Single DT", fontsize=16)

plt.subplot(322)
plot_predictions([tree_1], X, y, axes=axes, label="$h(x_1) = h_1(x_1)$", data_label="Train Data")
plt.ylabel("$y$", fontsize=12, rotation=0)
plt.title("Prediction of Ensemble", fontsize=16)

plt.subplot(323)
plot_predictions([tree_2], X, y2, axes=axes, label="$h_2(x_1)$", style="g-", data_style="k+", data_label="Residual Error")
plt.ylabel("$y - h_1(x_1)$", fontsize=16)

plt.subplot(324)
plot_predictions([tree_1, tree_2], X, y, axes=axes, label="$h(x_1) = h_1(x_1) + h_2(x_1)$")
plt.ylabel("$y$", fontsize=12, rotation=0)

plt.subplot(325)
plot_predictions([tree_3], X, y3, axes=axes, label="$h_3(x_1)$", style="g-", data_style="k+")
plt.ylabel("$y - h_1(x_1) - h_2(x_1)$", fontsize=12)
plt.xlabel("$x_1$", fontsize=12)

plt.subplot(326)
plot_predictions([tree_1, tree_2, tree_3], X, y, axes=axes, label="$h(x_1) = h_1(x_1) + h_2(x_1) + h_3(x_1)$")
plt.xlabel("$x_1$", fontsize=12)
plt.ylabel("$y$", fontsize=12, rotation=0)

plt.show()
```
<p align="center">
    <img src="Img/GBM.png" width="800"/>
</p>

์๊ฐํ ๊ฒฐ๊ณผ๋ฅผ ๋ณด๋ฉด, ์์ ์ด 1์์ 3์ผ๋ก ์ฆ๊ฐํ  ์๋ก Graident Boosting Model์ ์์ธก ๊ฐ์ด ์ ์  ์ค์  ๋ฐ์ดํฐ์ ์ fitting๋๋ ๊ฒ์ ํ์ธํ  ์ ์์ต๋๋ค.

**<span style="color:darkviolet">์ ๋ฆฌํ์๋ฉด Gradient Boosting์ ํ์ต ๋ฐฉ์์ ๋ค์๊ณผ ๊ฐ์ต๋๋ค.**   
1. ์ด๊ธฐ ๊ฐ์ผ๋ก ์์ ํจ์ ์ ์ฉ
2. Loss function์ ์ต์ํ ํ๋ gradient๋ฅผ ๊ตฌํจ
3. Gradient๋ฅผ $h^t$์ target์ผ๋ก ์ฌ์ฉํ์ฌ gradient๋ฅผ ๊ณ ๋ คํ ํ์ต ์งํ
4. ์ ์ ํ Learning rate $\alpha$๋ฅผ ๊ณ ๋ คํ ์ต์ข ๋ชจํ ์์ฑ
5. 2~4 ๊ณผ์  ๋ฐ๋ณต

<br/>

## <span style="background-color:#fff5b1"> [์คํ 6] Learning Rate์ ๋ฐ๋ฅธ Gradient Boosting์ ์ฑ๋ฅ ๋ณํ
์ด๋ ํ์ต ๋ฐฉ์์์ **<span style="color:maroon">"์ ์ ํ Learning rate"๋ฅผ ์ค์ ํ๋ ๊ธฐ์ค์ ๋ฌด์์ผ๊น์?** ์์ ์์๋ ์์๋ก 1e-2 ๊ฐ์ ์ฌ์ฉํ์์ผ๋, **<span style="color:maroon">Loss๋ฅผ ์ต์ํ ํ๋ gradient๋ฅผ ์ฐพ๊ธฐ ์ํด์๋ learning rate ๊ฐ์ ์ ์ค์ ํด์ฃผ๋ ๊ฒ์ด ๋งค์ฐ ์ค์ํด๋ณด์๋๋ค.** ์ด๋ฅผ ์คํ์ผ๋ก ํ์ธํด๋ณด๊ฒ ์ต๋๋ค.   
```python
from sklearn.ensemble import GradientBoostingClassifier
```
```python
# Get a list of models to evaluate
def get_models():
    models = dict()
    # Define learning rates to explore
    for i in [1e-4, 1e-3, 1e-2, 1e-1, 1e-0]:
        lr = '%.4f' % i
        models[lr] = GradientBoostingClassifier(learning_rate=i)
    return models


def print_results_per_lr(models:dict, X, y):
    acc_score, auroc_score, lr_list = [], [], []
    for k, model in models.items():
        # Evaluate Gradient Boosting Ensemble Models per learning rate
        acc, auroc = evaluate_model(model, X, y)
        # Store the Results
        acc_score.append(acc)
        auroc_score.append(auroc)
        lr_list.append(k)
        
        # Print the performance along the way
        print('lr=%s' % (k))
        
    return acc_score, auroc_score, lr_list
```
```python
# Define Dataset
X, y = get_dataset()

# Define Models
GBM = get_models()

# Evaluate Models
gbm_acc, gbm_auroc, gbm_lr = print_results_per_lr(GBM, X, y)
```

### Results
```
Accuracy: 0.74 (0.05)   
AUROC: 0.80 (0.05)   
lr=0.0001    

Accuracy: 0.75 (0.05)    
AUROC: 0.82 (0.05)    
lr=0.0010   

Accuracy: 0.81 (0.04)   
AUROC: 0.90 (0.03)   
lr=0.0100   

Accuracy: 0.91 (0.02)   
AUROC: 0.97 (0.01)   
lr=0.1000   

Accuracy: 0.92 (0.02)   
AUROC: 0.97 (0.01)   
lr=1.0000   
```
<br/>

## <span style="background-color:#fff5b1"> [์คํ 6] ๊ฒฐ๊ณผ ํด์
```python
plt.figure(figsize=(10, 5))
plt.title("Gradient Boosting Learning Rate vs. Classification Accuracy")
plt.boxplot(gbm_acc, labels=gbm_lr, showmeans=True);
```

<p align="center">
    <img src="Img/gbm_lr_acc.png" width="500"/>
</p>

```python
plt.figure(figsize=(10, 5))
plt.title("Gradient Boosting Learning Rate vs. Classification AUROC")
plt.boxplot(gbm_auroc, labels=gbm_lr, showmeans=True);
```

<p align="center">
    <img src="Img/gbm_lr_auroc.png" width="500"/>
</p>

- ๊ฒฐ๊ณผ๋ฅผ ์๊ฐํํ Plot์ ๋ณด๋ฉด, **<span style="color:maroon">Learning Rate๊ฐ ์ฌ๋ผ๊ฐ ์๋ก Gradient Boosting์ ์ฑ๋ฅ์ด ํฅ์ํ๋ ๊ฒฝํฅ์ ํ์ธ**ํ  ์ ์์ต๋๋ค.
- ์ด๋ **<span style="color:maroon">Learning Rate๊ฐ ๋์ ์๋ก ๋น ๋ฅด๊ฒ ๋ชจ๋ธ์ Bias๋ฅผ ์ค์ฌ๋๊ฐ๊ธฐ ๋๋ฌธ**์๋๋ค. ๊ทธ๋ ๋ค๋ฉด learning rate์ ๋์ ์๋ก ๋ฌด์กฐ๊ฑด ์ข์ ๊ฒ์ผ๊น์? ๊ทธ๋ ์ง๋ ์์ต๋๋ค.
- ์ฌ์ค **<span style="color:maroon">์ผ๋ฐ์ ์ผ๋ก๋ 1e-3(0.001) ~ 1e-2(0.01) ์์ค์ ๋ฎ์ ๊ฐ์ ์ค์ ํ๋ ๊ฒ์ด ๋ณดํต**์๋๋ค. ๊ทธ ์ด์ ๋ **์์ ๊ฐ์ผ๋ก ์ค์ ํ์ฌ์ผ ์ธ๋ฐํ Model์ ์ป์ ์ ์๊ธฐ ๋๋ฌธ**์๋๋ค. **<span style="color:maroon">Learning Rate๊ฐ ๋์ผ๋ฉด ๋น ๋ฅด๊ฒ ๋ชจ๋ธ์ Bias๋ฅผ ์ค์ผ ์ ์๋ ํํธ Fitting ๊ณผ์ ์์ detailํ ๋ถ๋ถ์ ๋์น  ์ ์๋ค๋ Trade-off๋ฅผ ๊ฐ์**ํด์ผ ํ๋ ๊ฒ์๋๋ค.

<br/>

# <span style="color:purple">Boosting 2: <span style="color:crimson">CatBoost ๐ป
CatBoost๋ "Gradient Boosting with Categorical Features Suppeort", ์ฆ, ์ค๋ช ๋ณ์์ Category ํ์์ ๋ฐ์ดํฐ๊ฐ ํฌํจ๋์ด ์์ ๋ ์ ์ฉํ๊ฒ ์ฌ์ฉ๋๋ Gradient Boosting ๊ณ์ด ๋ฐฉ๋ฒ๋ก ์๋๋ค. CatBoost์ ์ ์๋ค์ ๊ธฐ์กด์ Gradient Boosting ๋ชจ๋ธ์ ๋ฌธ์ ์ ์ ์ง์ผ๋ฉด์ ๋ผ๋ฌธ์ ์๋ก ์ ์์ํ๋๋ฐ, ๋ชจ๋ธ์ ์์ฐจ์ ์ผ๋ก ์๋ฐ์ดํธ ํ๋ ๋ฐ ์์ด **Greedy Manner**๋ฅผ ์ด์ฉํ๋ค๋ ๋ฌธ์ ๋ฅผ ์ง์ ํฉ๋๋ค. ์ด๊ฒ์ด **<span style="color:crimson">Inference์์ ํ์ฉ๋์ด์ผ ํ  ๋ฐ์ดํฐ๋ฅผ Train ๋จ๊ณ์์ ์ด์ฉํ๊ณ  ์๋ค๋ ๋ฌธ์ ๋ฅผ ์ง์ **ํ ๊ฒ์ด์ฃ .   

์์ Gradient Boosting์์ t๋ฒ์งธ Boosting ๋ชจํ $F_t$๋ฅผ ๋ง๋ค ๋์๋, t-1๋ฒ์งธ๊น์ง ๋์ ๋ ๋ชจํ $F_{t-1}$์ ํ์ฌ ์์ ์ ๋ชจํ $h_t$๋ฅผ ๋ํด์ฃผ๋ ๋ฐฉ์์ ์ด์ฉํ๋ค๊ณ  ํ์์ต๋๋ค. ๋ฐ๋ผ์ ์๋์ ๊ฐ์ด t ์์ ์ Boosting ๋ชจํ์ ๊ตฌํ  ์ ์๋ ๊ฒ์ด์ฃ . 
$$F_t(x) = F_{t-1}(x) + \alpha h_t(x)$$
์ด๋ $h_t$๋, t-1 ์์ ๊น์ง์ ๋์  ๋ชจ๋ธ $F_{t-1}$์ $h_t$๋ฅผ ๋ํ์์ ๋ ์ถ์ ๋๋ ์์ธก ๊ฐ์ ์ค์  ๊ฐ๊ณผ ๋น๊ตํ์ ๋ ๊ทธ Loss๊ฐ ์ต์ํ ๋๋ ํจ์๋ฅผ ์ฐพ์ต๋๋ค. ์ด๋ ์๋์ ๊ฐ์ ์์ผ๋ก ๋ํ๋ผ ์ ์์ต๋๋ค.
$$h_t = arg\ \underset{h \in H}{min}\mathcal{L}(F_{t-1}+h) = arg \ \underset {h \in H}{min} \ \mathbb{E}L(y)F_{t-1}(x)+h(x)$$   

์ด๋ $h_t$๋ฅผ ๊ทผ์ฌํ๋ ๋ฐ๋, t ์์ ์์์ gradient์ ๋ฐ๋ ๋ฐฉํฅ์ผ๋ก ์ด๋ฅผ ์ถ์ ํฉ๋๋ค. ์ฃผ๋ก Least Square Approximation์ ์ฌ์ฉํ๋๋ฐ, ์ด๋ ์๋์ ๊ฐ์ต๋๋ค.

$$h_t = arg \ \underset{h\in H}{min} \ \mathbb{E}(-g_t(x, y)-h(x))^2 \\
g_t(x, y) := \frac{\partial L(y,s)}{\partial F_{t-1}(x)}$$

๋ค์ ๋งํด, negative gradient = $-g_t(x, y)$์์ ํ์ฌ h(x)๋ผ๋ ํจ์๋ฅผ ์ถ์ ํ์ ๋, ๊ทธ ์ฐจ์ด์ ๋ํ Expectation์ด ์ต์ํ ๋๋ t๋ฒ์งธ ์์ ์์์ tree ๋ชจํ์ ๋ง๋๋ ๊ฒ์ด Gradient Boosting ๋ชจํ์ด์ฃ .   

์ด์ CaBoost์ ์ ์๋ค์, **<span style="color:crimson">๊ธฐ๋ณธ์ ์ธ Gradient Boosting ๊ณ์ด์ ๋ฐฉ๋ฒ๋ก ๋ค์ด ๊ฐ์ง๋ 2๊ฐ์ง ๋ฌธ์ ๋ฅผ ์ ๊ธฐ**ํฉ๋๋ค.  

## Problems of Gradient Boosting 1: <span style="color:crimson">Prediction Shift
๋จผ์  $h_t$๋ฅผ ์ถ์ ํ๋ ๋ฐ ์์ด, ๋ชจ๋  ๋ฐ์ดํฐ์์ ๋ํ ๊ธฐ๋๊ฐ์ ์ต์ํ ํ๋ ๊ฒ์ ์ ํํ ๊ด์ธก์น ๊ฐ์๋ฅผ ๊ฐ์ง๋ ๋ฐ์ดํฐ์์๋ ๋ถ๊ฐ๋ฅํ ์ผ์๋๋ค. ๋ฐ๋ผ์ ์๋์ ๊ฐ์ด Training Dataset์ ๋ํ ํ๊ท ์ฒ ์ฐจ์ด๋ก ๊ทผ์ฌํ๊ฒ ๋์ฃ .  
    
$$h_t = arg \underset{h \in H}{min}\mathbb{E}(-g_t(x, y)-h(x))^2 \approx \frac{1}{n}\sum^{n}{k=1}(-g_t(x,y)-h(x))^2 \\  Training \ Dataset: \mathcal{D}=(x_k, y_k)_{k=1,...,n} \ where \ x_k=(x^1_k, ..., x^m_k), \quad y_k \in \mathbb{R}$$
    
๋ฐ๋ก ์ด๋ฌํ ์ง์ ์์, ํ์ต ๋ฐ์ดํฐ์์ $x_k$๊ฐ ์ฃผ์ด์ก์ ๋, ์ง๊ธ๊น์ง ์ฐ๋ฆฌ๊ฐ ๋ง๋ค์๋ ๋์ ๋ Boosting ๋ชจํ์ ๊ฐ๊ณผ Test Example $x$๊ฐ ์ฃผ์ด์ก์ ๋ ๋์ ๋ Boosting ๋ชจํ์ ๊ฐ์ด ๋ค๋ฅด๋ค๋ ๋ฌธ์ ๊ฐ ๋ฐ์ํฉ๋๋ค. ๋ค์ ๋งํด, **<span style="color:crimson">train example $x_k$๊ฐ ์ฃผ์ด์ก์ ๋์ gradient์, $x$๊ฐ ์ฃผ์ด์ก์ ๋์ test example์์์ gradient์ conditional distribution์ด ๋ค๋ฅธ ๊ฒ**์๋๋ค.   
$$F_{t-1}(x_k)|x_k โ  F_{t-1}(x)|x$$
์ด ๋ ๊ฐ์ง๊ฐ ๊ฐ์์ผ ๋ชจ๋ธ๋ง์ ์ ํฉ์ฑ์ด ํ๋ณด ๋๋๋ฐ, ํ์ต ๋ฐ์ดํฐ์ ๋ํ ๋์  ํจ์์ ์กฐ๊ฑด๋ถ ํ๋ฅ ๊ณผ ๊ฒ์ฆ์ฉ ๋ฐ์ดํฐ์ ๋ํ ์กฐ๊ฑด๋ถ ํ๋ฅ ์ด ๋ค๋ฅธ ๊ฒ์ด ๋ฐ๋ก ์ฒซ ๋ฒ์งธ Issue์๋๋ค. ์ด๋ฅผ Prediction Shift๋ผ ํ๊ณ , ์ด๋ ๊ฒ **<span style="color:crimson">ํธํฅ๋ $h_t$๋ฅผ $F_t$๋ฅผ ๋ง๋๋ ๋ฐ ์ฌ์ฉํ๋ฉด, ๊ฒฐ๊ตญ $F_t$์ ์ผ๋ฐํ ์ฑ๋ฅ์ด ๋ฌธ์ ๊ฐ ๋๋ค๋ ์ ์ ์ง์ ๊ฒ์ด์ฃ . ์ค์ ๋ก Gradient Boosting ๋ชจ๋ธ์ Overfitting์ ๋ฌธ์ **๋ฅผ ์๊ณ  ์์ต๋๋ค.   

<br/>

## <span style="color:crimson">Solution 1: Ordered Boosting
์ด๋ฅผ ํด๊ฒฐํ๊ธฐ ์ํด CatBoost์์ ์ ์ํ ๋ฐฉ๋ฒ์ด ๋ฐ๋ก **<span style="color:crimson">Ordered Boosting**์๋๋ค. 'Ordered'๋ผ๋ ๋ง์ด ๋ค์ด๊ฐ ์ด์ ๋, ๋ณ์์ ๋ํด ๋ฌด์์ permutation๋ฅผ ์ํํ์ฌ ์์ด์ ๋ง๋ค๊ณ , ์์ฐจ์ ์ผ๋ก ์์ฐจ๋ฅผ ๊ณ์ฐํ๋ฉฐ tree๋ฅผ ํ์ต์ํค๊ธฐ ๋๋ฌธ์๋๋ค.   

๋ง์ผ 9๊ฐ์ ๋ณ์๊ฐ ์๋ค๊ณ  ๊ฐ์ ํ์๋ฉด, ๋ฐฉ๋ฒ๋ก ์ ์๋์ ๊ฐ์ด ๋์ํ ํ  ์ ์์ต๋๋ค.   
<p align="center">
    <img src="Img/ordered_boosting.PNG" width="800"/>
</p>

$M_5^{t-1}$์ 5๋ฒ์งธ ๋ฐ์ดํฐ๊น์ง๋ง์ ์ฌ์ฉํด ๋ง๋ค์ด๋ธ ๋ชจ๋ธ์ด๊ณ , $M_6^{t-1}$์ 6๋ฒ์งธ ๋ฐ์ดํฐ๊น์ง๋ง์ ์ฌ์ฉํด ๋ง๋ค์ด๋ธ ๋ชจ๋ธ์ด๊ฒ ์ฃ . <span style="color:crimson">์ค์ํ ๊ฒ์ **<span style="color:crimson">7๋ฒ์งธ ๋ฐ์ดํฐ์ ๋ํด ์์ฐจ(residual)๋ฅผ ๊ตฌํ  ๋, $M_7^{t-1}$๋ฅผ ์ด์ฉํ๋ ๊ฒ์ด ์๋๋ผ, $M_6^{t-1}$๋ฅผ ์ด์ฉํด์ผ ํ๋ค๋ ๊ฒ**</span>์๋๋ค. ์ ๊ทธ๋ด๊น์? **<span style="color:crimson">$M_6^{t-1}$๋ฅผ ๋ง๋ค ๋ 7๋ฒ์งธ ๋ฐ์ดํฐ, $x_3$์ ํ ๋ฒ๋ ์ฌ์ฉ๋ ์ ์ด ์์ฃ . ๊ทธ๋ ๊ธฐ์ inference ๋์ ๋์ผํ ํ๊ฒฝ์ ์กฐ์ฑ**ํ  ์ ์๋ค๋ ๊ฒ์๋๋ค. ์ด๋ฅผ ํตํด์ Prediction Shift ๋ฌธ์ ๋ฅผ ํด๊ฒฐํ๊ณ  ์์ต๋๋ค.   

<br/>

## <span style="background-color:#fff5b1"> [์คํ 7] CatBoost vs. Gradient Boosting: in terms of performance
์ฌ์ค Ensemble Learning์ ์ฌ๋ฌ ๋ชจ๋ธ์ ์์ธก ๊ฐ์ ํฉ์น๋ ์ทจํฉํ๋ ๋ฐฉ์์ด๊ธฐ ๋๋ฌธ์, Single Model๋ณด๋ค๋ Overfitting์ ๋ฐฉ์งํ  ์ ์๋ ๊ฐ๋ฅ์ฑ์ด ๋์ต๋๋ค. ๊ทธ๋ผ์๋ Deisicion Tree๋ฅผ ๊ธฐ๋ฐ์ผ๋ก ํ๋ ์๊ณ ๋ฆฌ์ฆ์ ๊ฒฝ์ฐ์๋ ์ฌ์ ํ Overfitting์ ๋ฌธ์ ๋ฅผ ํผํด๊ฐ๊ธฐ ํ๋ค๋ฉฐ, Gradient Boosting ๊ณ์ด ๋ชจ๋ธ๋ค์ ๊ฒฝ์ฐ ์์  ์ด์ ์์ ๊ณผ์ ํฉ์ ๊ฒฝํฅ์ ๋ณด์ด๊ณ  ์์ค๋๋ค.   

**<span style="color:maroon">CatBoost๋ ์ ๋ง GBM๊ณผ๋ ๋ค๋ฅด๊ฒ Overfitting์ ์ํํ์์๊น์? ๊ทธ๋ ๋ค๋ฉด Test ๊ฒฐ๊ณผ CatBoost์ ์ฑ๋ฅ์ด ๋ ๋์ ๊ฒ์๋๋ค.** CatBoost์ ๊ธฐ๋ณธ์  ์ฑ๋ฅ์ ์คํ์ผ๋ก ํ์ธํด๋ณด๊ฒ ์ต๋๋ค.   

```python
from catboost import CatBoostClassifier
```
```python
# Define Dataset
X, y = get_dataset()

# Gradient Boosting Classifier
gbm_clf = GradientBoostingClassifier(random_state=2022, learning_rate=1e-2)

# CatBoost Classifier
cat_clf = CatBoostClassifier(random_seed=2022, learning_rate=1e-2)
```
- GBM๊ณผ CatBoost๋ ๋์ผํ Hyperparameter๋ฅผ ์ฌ์ฉํ๋ฉฐ, learning_rate๋ 1e-2๋ก ์ผ๋ฐ์ ์ผ๋ก ์์ฃผ ์ฌ์ฉ๋๋ ์์ ๊ฐ์ ์์๋ก ์ค์ ํด์ค๋๋ค. learning rate๊ฐ 1e-2์ธ ์ด์ ๋ **<span style="background-color:#fff5b1">[์คํ 6]** ์์ ํ์ธํ์์ต๋๋ค.   

```python
# Evaluate GBM
acc_gbm, auroc_gbm = evaluate_model(model=gbm_clf, X=X, y=y)

# Evaluate CatBoost
acc_cat, auroc_cat = evaluate_model(model=cat_clf, X=X, y=y)
```

### Results
|                       | __Gradient Boosting__| __CatBoost__ |
|:---------------------:|:--------------------:|:------------:|
|__Mean Accuracy (std)__| 0.81 (0.04)          | **0.95 (0.02)**  |
|   __Mean AUROC (std)__|   0.90 (0.03)        |  **0.99 (0.01)** |

<br/>

### <span style="background-color:#fff5b1"> [์คํ 7] ๊ฒฐ๊ณผ ํด์
- ์คํ ๊ฒฐ๊ณผ, **<span style="color:maroon">CatBoost๊ฐ ๊ด๋ชฉํ  ๋งํ ์ฑ๋ฅ์ ๋ด๋ฉฐ, ์ฑ๋ฅ ์ธก๋ฉด์์ ์ฐ์์ฑ์ ์์ฆ**ํ์์ต๋๋ค. Accuracy ์ธก๋ฉด์์ 14%, AUROC ์ธก๋ฉด์์๋ 9% ๊ฐ๋ ๋์ ์ฑ๋ฅ์ ๊ธฐ๋กํ ๊ฒ์๋๋ค.
- ๋๋ถ์ด **<span style="color:maroon">๋ชจ๋ธ๋ค์ ์์ธก ๊ฐ ํธ์ฐจ๋ Auuracy์ AUROC ์ธก๋ฉด์์ ๋ชจ๋ ๋ ๋ฎ์ ๊ฐ์ ๊ธฐ๋กํ๋ฉฐ, Bias๋ฟ ์๋๋ผ Variance๋ฅผ ๋ฎ์ถ ์ด์์ ์ธ ๋ชจ๋ธ๋ก์ ์ฑ๋ฅ์ ์์ฆ**ํ์ต๋๋ค.
- ์ด๋ Inference ์ ํ์ฉ๋์ด์ผ ํ  ๋ฐ์ดํฐ๋ฅผ ํ์ต ์ ๋ชจ๋ธ ์๋ฐ์ดํธ ๊ณผ์ ์์ ์ง์์ ์ผ๋ก ์ฌ์ฉํ๋ Gradient Boosting๊ณผ๋ ๋ค๋ฅด๊ฒ, **<span style="color:maroon">CatBoost์์๋ t ์์ ์ ์์ฐจ๋ฅผ ๊ณ์ฐํ๋ ๊ณผ์ ์์ t-1 ์์ ์ ๋์  Boosting ๋ชจํ์ ํ์ฉํ๋ฉด์, Inference ๋์ ๋์ผํ ํ๊ฒฝ์ ์กฐ์ฑํ์๊ธฐ ๋๋ฌธ์๋๋ค. ์ฆ, Generalization ์ฑ๋ฅ์ด ํ๋ณด**๋ ๊ฒ์ด์ฃ .

<br/>

## <span style="background-color:#fff5b1"> [์คํ 8] CatBoost vs. Gradient Boosting: in terms of training time
์์ CatBoost๋ random permutation์ ํ์ต ๊ณผ์ ๋ง๋ค ์ค์ํ๊ณ  ์์ฐจ์ ์ผ๋ก residual์ ๊ณ์ฐํ๋ฉฐ ์ง์์ ์ธ update๋ฅผ ์ํํ๋ค๊ณ  ํ์์ต๋๋ค. ์ด๋ก์จ Overfitting์ ๋ฐฉ์งํ๊ณ  Testing ์ฑ๋ฅ์ ๋์ผ ์ ์์ง๋ง, **<span style="color:maroon">ํ์ต ์ ์์ ์๊ฐ์ด ๋งค์ฐ ์ค๋ ๊ฑธ๋ฆฌ์ง๋ ์์๊น์?** ์ด๋ฅผ ์ง์  ์คํ์ ํตํด ๋น๊ตํด๋ณด๊ฒ ์ต๋๋ค.   

```python
def evaluate_model_time(model, X, y, task: str):
    training_time = dict()
    
    start = time.time()
    
    # Define the evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=2022)
    # Evaluate model and collect the results
    acc = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    roc_auc = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    training_time[task] = np.round(time.time() - start, 3)
    
    # Report Performance
    print("Accuracy: %.2f (%.2f)" % (np.mean(acc), np.std(acc)))
    print("AUROC: %.2f (%.2f)" % (np.mean(roc_auc), np.std(roc_auc)))
    
    # Report Training Time
    print(f"Training Time: {training_time[task]}s")
    
    return acc, roc_auc
```
```python
acc_gbm, auroc_gbm = evaluate_model_time(model=gbm_clf, X=X, y=y, task="Gradient Boosting")

acc_cat, auroc_cat = evaluate_model_time(model=cat_clf, X=X, y=y, task="CatBoost")
```

### Results
|                       | __Gradient Boosting__| __CatBoost__ |
|:---------------------:|:--------------------:|:------------:|
| __Training Time__     |  **2.192 sec**           |   73.247 sec   |
|__Mean Accuracy (std)__| 0.81 (0.04)          | **0.95 (0.02)**  |
|   __Mean AUROC (std)__|   0.90 (0.03)        |  **0.99 (0.01**) |

<br/>

### <span style="background-color:#fff5b1"> [์คํ 8] ๊ฒฐ๊ณผ ํด์
- ์คํ ๊ฒฐ๊ณผ **<span style="color:maroon">CatBoost์ ํ์ต ์๊ฐ์ด ์ฝ 33% ์ด์ ์ค๋ ๊ฑธ๋ฆฐ ๊ฒ์ ํ์ธ**ํ์์ต๋๋ค. 
- ์ด๋ ์์ ์ธ๊ธํ ๋ฐ์ ๊ฐ์ด **<span style="color:maroon">CatBoost๊ฐ ํ์ต ๋จ๊ณ๋ง๋ค Random Permutation์ ์ํ ํ ์์ฐจ์ ์ผ๋ก ์์ฐจ๋ฅผ ๊ณ์ฐํ๋ ํ์ต ๋ฐฉ์์ ํตํด ๋ชจ๋ธ์ ์๋ฐ์ดํธํ๊ธฐ ๋๋ฌธ**์๋๋ค.
- ์ฆ, CatBoost๋ ์ฑ๋ฅ์ ๋์ง๋ง ํ์ต ์๊ฐ์ด ๋งค์ฐ ์ค๋ ๊ฑธ๋ฆฌ๊ธฐ์, ๋ ์ฌ์ด์ Trade Off๋ฅผ ๊ณ ๋ คํด์ผ ํ๋ ๋ชจ๋ธ์๋๋ค.


<br/>

## Problems of Gradient Boosting 2: <span style="color:crimson">Target Leakage (in Target Statistics)   
> **Target Statistics (TS)**   
> Categorical ๋ณ์๋ค์ Numerical Value๋ก ๋ฐ๊พธ๋ ๋ฐฉ๋ฒ๋ก ์๋๋ค. ์ผ๋ฐ์ ์ผ๋ก ๊ฐ์ฅ ๋ฒ์ฉ์ ์ผ๋ก ์ฌ์ฉ๋๋ ๊ฒ์ Mean-Encoding/Target-Encoding์ด๋ผ ๋ถ๋ฆฌ๋ ๋ฐฉ๋ฒ๋ก ์ผ๋ก, ํ์ต ๋ฐ์ดํฐ์์ **Categorical Features๋ฅผ Target Value์ ํ๊ท ์ผ๋ก ๋์ฒด**ํ๋ ๋ฐฉ๋ฒ๋ก ์ ์ผ์ปซ์ต๋๋ค.   

๊ธฐ์กด Gradient Boosting ๋ชจ๋ธ์ ๋ ๋ฒ์งธ ์ด์๋ Target Leakage์๋๋ค. **<span style="color:crimson">ํ์ต ๋ฐ์ดํฐ์ Target(์ ๋ต) ์ ๋ณด๋ฅผ ์๋ ฅ ๋ณ์, ์ฆ ๊ฐ์ฒด์ ์์ฑ์ ์ ์ํ๋ ๋ฐ ์ด๋ฏธ ์ฌ์ฉ๋๊ณ  ์๋ค๋ ๋ฌธ์ ๋ฅผ ์ ๊ธฐ**ํ ๊ฒ์๋๋ค. ์๋ ๋จธ์ ๋ฌ๋์ ๋ชฉ์ ์ ์๋ ฅ ๋ณ์ $x$๊ฐ ์ฃผ์ด์ก์ ๋ $y$๋ฅผ ์ถ์ ํ  ์ ์๋ ํจ์ $f(\cdot)$๋ฅผ ์ฐพ๋ ๊ฒ์ด์ฃ . ๊ทธ๋ฐ๋ฐ ์ผ๋ฐ์ ์ธ Boositng์์์ Target Statistics ๋ฐฉ์์์๋ ์ค์  $y$ ๊ฐ์ด $\hat y$์ ๊ณ์ฐํ๋ ๋ฐ ์ฌ์ฉ๋๊ณ  ์๋ค๋ ๋ฌธ์ ๊ฐ ์์ต๋๋ค. ์ฆ, $x$๋ก $y$๋ฅผ ์ถ์ ํ๋ค๋ ๊ธฐ๋ณธ ๊ฐ์ ์ ์๋ฐฐํ ์ฑ target ์ ๋ณด๊ฐ ์์ด๋๊ฐ๊ณ  ์๋ค๋ ๊ฒ์ด์ฃ .  

๋ฐ๋ผ์ CatBoost์ ์ ์๋ค์ **Target Statistics์์ ํ์ํ ๋ฐ๋์งํ ์์ฑ**์ ๋ํ์ฌ ์ง๋ฌธ์ ๋์ง๊ณ , ๋๋ฆ์ 2๊ฐ์ง ์์ฑ์ ์ ์ํฉ๋๋ค.
$$Property\ 1 \quad \mathbb{E}(\hat x^i | y=v) = \mathbb{E}(\hat x^i_k | y_k=v), where \ (x_k, y_k) \ is \ the \ k-th \ training \ example.$$
$$Property\ 2 \quad Effective\ usage\ of\ all\ training\ data\ for\ calculating\ TS\ features\ and\ for\ learning\ a\ model$$   

**Property 1**์ $y$๊ฐ $v$๋ผ๋ ๊ฐ์ ๊ฐ์ง ๋, $i$๋ฒ์งธ ์์ฑ์ ํด๋นํ๋ Expectation๊ณผ, k๋ฒ์งธ training example์ ๋ํด์ $y_k$๊ฐ $v$์ผ ๋์ Expectation์ด ๊ฐ์์ผ ํ๋ค๋ ์๋ฏธ์๋๋ค. ์ฝ๊ฒ ๋งํด, **$y$๊ฐ $v$๋ผ๋ ๊ฐ์ ๊ฐ์ง ๋ Train ๋ฐ์ดํฐ์ Test ๋ฐ์ดํฐ์ ๋ํด์ Target Statistics์ ๊ธฐ๋๊ฐ์ด ๋ชจ๋ ๋์ผํด์ผ ํ๋ค๋ ๊ฒ**์๋๋ค.   

<br/>

**Property 2**๋ ๊ฐ๊ธ์ ์ด๋ฉด **Target Statistics๋ฅผ ๊ณ์ฐํ๋ ๋ฐ ์์ด ๋ชจ๋  Dataset์ ํจ๊ณผ์ ์ผ๋ก ์ฌ์ฉํ๋ผ๋ ์๋ฏธ**์๋๋ค. ๊ฒฐ๊ตญ ๊ฐ๋ฅํ ๋ง์ ํ์ต ๋ฐ์ดํฐ๋ฅผ ์ฌ์ฉํด์ TS๋ฅผ ๊ณ์ฐํ๋ผ๋ ๊ฒ์ด์ฃ .

<br/>

## <span style="color:crimson">Solution 2: Ordered Target Statistics
์์ ๊ฐ์ ์์ฑ์ ๋ง์กฑ์ํค๊ธฐ ์ํด์, ์ ์๋ค์ด ์ ์ํ Solution์ด ๋ฐ๋ก Ordered Target Statistics์๋๋ค. ์ฌ๊ธฐ์ 'Ordered'๋ผ ํจ์, ์์ ์ธ๊ธํ ๊ฒ๊ณผ ๋ง์ฐฌ๊ฐ์ง๋ก, **<span style="color:crimson">๊ฐ์ฒด๋ฅผ ๋ฌด์์๋ก permutation ์ํจ ํ, Artificial Time, ์ฆ ๊ฐ์์ ์๊ฐ ๊ฐ๋์ ๋ถ์ฌํ๊ธฐ ๋๋ฌธ**์๋๋ค. ์ด๋ Ordered Boosting๊ณผ ๋ง์ฐฌ๊ฐ์ง๋ก, **$x_k$์ ๋ํ TS๋ฅผ ๊ณ์ฐํ  ๋๋ $x_k$์ ์ด์  ์ ๋ณด๋ง์ ํ์ฉ**ํฉ๋๋ค. ๊ทธ๋ ๋ค๋ฉด TS๋ฅผ ์ํด ํ์ํ subset $\mathcal{D_k}$๋ ์๋์ ๊ฐ์ด ํ๊ธฐ๋๊ฒ ์ฃ .
    
$$\mathcal{D}_k \subset \mathcal{D} \ \backslash \begin{Bmatrix}
x_k
\end{Bmatrix} \ excluding \ x_k$$

$$
\mathcal{D}_k = \begin{Bmatrix}
x_j : \sigma (j) < \sigma (k)
\end{Bmatrix}
$$

                       
์ฌ๊ธฐ์ $\sigma$๋ random permutation์ ์๋ฏธํ๋ parameter์๋๋ค. ์ด๋ฅผ ๋ฐํ์ผ๋ก Categorical ๋ณ์ $x^i$์์ k๋ฒ์งธ ๊ฐ์ฒด $x^i_k$๋ ๋ค์๊ณผ ๊ฐ์ด ๊ตฌํ  ์ ์์ต๋๋ค.  
                         
$$\hat{x}^i_k = \frac{\sum_{x_j \in \mathcal{D}_k} 1 \begin{Bmatrix} x^i_j = x^i_k \end{Bmatrix} \cdot y_j + ap}{\sum_{x_j \in \mathcal{D}_k} 1 \begin{Bmatrix} x^i_j = x^i_k \end{Bmatrix} + a}$$

- ๋จผ์  ๋ถ์์ ๋ถ๋ชจ์ ๊ณตํต์ ์ผ๋ก ๋ค์ด๊ฐ term์ธ $\sum_{x_j \in \mathcal{D}_k} \ {1} \begin{Bmatrix} x^i_j = x^i_k \end{Bmatrix}$๋ ๋ฌด์จ ์๋ฏธ์ผ๊น์? **k๋ฒ์งธ ๊ด์ธก์น $x_k$ ์ง์ ๊น์ง์ ๋ชจ๋  ๋ฐ์ดํฐ์ ๋ํ์ฌ, $x_k$์ ๋์ผํ ์นดํ๊ณ ๋ฆฌ ๊ฐ์ ๊ฐ์ง๋ ๊ด์ธก์น์ ๊ฐ์๋ฅผ ์๋ฏธ**ํฉ๋๋ค.
                         
- $\sum_{x_j \in \mathcal{D}_k} \ {1} \begin{Bmatrix} x^i_j = x^i_k \end{Bmatrix} \cdot y_j$๋, **k๋ฒ์งธ ๊ด์ธก์น ์ง์ ๊น์ง์ ๋ชจ๋  ๋ฐ์ดํฐ์ ๋ํ์ฌ, $x_k$์ ๋์ผํ ์นดํ๊ณ ๋ฆฌ ๊ฐ์ ๊ฐ์ง๋ ๊ด์ธก์น์ Target ๊ฐ**์ ์๋ฏธํ๊ฒ ์ฃ .
- ์ด๋ **$a$๋ Permutation์ ๋ํ Hyperparameter**์ด๊ณ , Ordered Boosting์์๋ ํจ๊ป ์ฌ์ฉ๋๋ ๊ฐ์๋๋ค.
- p๋ **k๋ฒ์งธ ๊ด์ธก์น ์ง์ ๊น์ง์ ๋ชจ๋  ๋ฐ์ดํฐ์ ๋ํ์ฌ, ํน์  Target์ด ๋ํ๋  ์ ํ ํ๋ฅ **์ ์๋ฏธํฉ๋๋ค.

<br/>

## <span style="background-color:#fff5b1">  [์คํ 9] Mean Target Encoding vs. Ordered Target Statistics
CatBoost๋ ์ด๋ฆ์์๋ ์ ์ ์๋ฏ์ด Categorical ๋ณ์๋ฅผ ๋ค๋ฃธ์ ์์ด ์ต์ ํ๋ ์๊ณ ๋ฆฌ์ฆ์๋๋ค. ์ด๋ ์์ ์ธ๊ธํ Target Mean ๋ฐฉ์์ Target Leakage ๋ฌธ์ ๋ก ์ธํ Overfitting ํ์์ด ์ผ์ด๋๋ค๊ณ  ์ง์ ๋ฐ ์์ต๋๋ค. ๋ฐ๋ผ์ CatBoost๋ Ordered TS๋ฅผ ํตํด Target Leakage๋ฅผ ํด๊ฒฐํ๊ณ ์ ํฉ๋๋ค.   

์ด๋ ๋ณธ ํํ ๋ฆฌ์ผ์์๋ **<span style="color:maroon">Binary Classification Task์ ์์ด, Ordered Target Statistics์ Target Mean Encoding ๋ชจ๋ ์ง์  Scratch๋ก ๊ตฌํํด๋ณด๊ณ  ๊ฒฐ๊ณผ๋ฅผ ๋น๊ต**ํด๋ณด๊ณ ์ ํฉ๋๋ค.   

ํ์ฉํ๋ ๋ฐ์ดํฐ์์ titanic dataset ์ผ๋ถ์ด๋ฉฐ, Categorical ๋ณ์๋ฅผ ๋ค๋ฃจ๊ธฐ ์ํด 'Embarked' Column๋ง์ ํ์ฉํฉ๋๋ค. **<span style="color:maroon"> ๋ณธ ํํ ๋ฆฌ์ผ์ ๋ชฉ์ **์ titanic ๋ฐ์ดํฐ์์์ ๋์ ๋ถ๋ฅ ์ฑ๋ฅ์ ๋ฌ์ฑํ๊ธฐ ์ํ ๋ถ์์ด ์๋๋ผ, **<span style="color:maroon"> Categorical ๋ณ์ ์ฒ๋ฆฌ ๋ฐฉ์์ ๋น๊ตํ๊ธฐ ์ํจ**์ด๊ธฐ ๋๋ฌธ์๋๋ค.

```python
from catboost.datasets import titanic

# Loade Dataset
df, _ = titanic()
df = df[['Embarked', 'Survived']].dropna().reset_index(drop=True)

# Split Dataset into train/valid/test
train_idx, test_idx = train_test_split(np.arange(df.shape[0]), train_size=.8, random_state=2022)

df_train = df.iloc[train_idx].reset_index(drop=True)
df_test = df.iloc[test_idx].reset_index(drop=True)
```
- ํ์ฉํ๋ ๋ฐ์ดํฐ์์ titanic dataset ์ผ๋ถ์ด๋ฉฐ, Categorical ๋ณ์๋ฅผ ๋ค๋ฃจ๊ธฐ ์ํด 'Embarked' Column๋ง์ ํ์ฉํฉ๋๋ค. **๋ณธ ํํ ๋ฆฌ์ผ์ ๋ชฉ์ **์ titanic ๋ฐ์ดํฐ์์์ ๋์ ๋ถ๋ฅ ์ฑ๋ฅ์ ๋ฌ์ฑํ๊ธฐ ์ํ ๋ถ์์ด ์๋๋ผ, **Categorical ๋ณ์ ์ฒ๋ฆฌ ๋ฐฉ์์ ๋น๊ตํ๊ธฐ ์ํจ**์ด๊ธฐ ๋๋ฌธ์๋๋ค.
- y๋ 'Survived'์ด๋ฉฐ, ์์กด ์ฌ๋ถ๋ฅผ ๊ตฌ๋ถํ๋ Binary Classification Task์๋๋ค.
- ์ด๋ ๋ณธ ํํ ๋ฆฌ์ผ์์๋ **Binary class๋ฅผ ๊ฐ์ง๋ Target์ ์ ๋ณด๋ฅผ ํ์ฉํ์ฌ Target Statistics ๊ฐ์ ๋ง๋๋ ๋ฐฉ๋ฒ์ ์ง์  ๊ตฌํ**ํด๋ณผ ๊ฒ์๋๋ค.


### <span style="color:maroon">1. Target Mean Encoding
```python
def Mean_TS_for_binary_clf(df: pd.DataFrame, X_col: str, y_col:str):
    Mean_TS_encode = df[y_col].groupby(df[X_col]).agg(['mean'])['mean']
    
    df.loc[:, 'TS'] = df[X_col].map(Mean_TS_encode)
    
    return df
```
- Binary Classification Task๋ฅผ ํ๊ณ ์ ํ  ๋, Target Mean Encoding ๋ฐฉ์์ ์์ ๊ฐ์ด ๊ตฌํ๋  ๊ฒ์๋๋ค.

#### <span style="color:maroon"> Target Mean Encoding: TS in Train Data
```python
df_train_TS = Mean_TS_for_binary_clf(df=df_train, X_col='Embarked', y_col='Survived')
df_train_TS
```
```
    Embarked	Survived	TS
761	    C	        1	0.550725
546	    C	        1	0.550725
450	    S	        0	0.349421
135	    S	        1	0.349421
58	    S	        1	0.349421
...	    ...	       ...	    ...
240	    Q	        1	0.327273
689	    S	        1	0.349421
624	    S	        0	0.349421
173	    C	        0	0.550725
220	    S	        0	0.349421
```
```python
df_test_TS = Mean_TS_for_binary_clf(df=df_test, X_col='Embarked', y_col='Survived')
df_test_TS
```
```
    Embarked	Survived	TS
747	    S	        0	0.285714
178	    S	        0	0.285714
784	    S	        0	0.285714
159	    S	        0	0.285714
337	    S	        1	0.285714
...	    ...	       ...	...
233	    S	        0	0.285714
80	    S	        1	0.285714
453	    S	        0	0.285714
136	    S	        0	0.285714
627	    S	        0	0.285714
```
- Mean Target Encoding์ ์ ์ฉํ  ์ Train data์์ `Embarked` Column์ด ๊ฐ์ง๋ Target Statistics ๊ฐ์ ์์ ๊ฐ์ต๋๋ค.
- ๋ชจ๋  ํ์ต ๋ฐ์ดํฐ์ ๋ํด์, $encoding = (count\ of\ target\ 1)/(total\ occurence)$์ ๋ฐฉ์์ผ๋ก ๊ฐ์ด ๊ณ์ฐ๋๊ธฐ์, ํ๋์ ์นดํ๊ณ ๋ฆฌ๋ Target ๊ฐ๊ณผ ์๊ด ์์ด ๋ชจ๋ ๋์ผํ TS๋ก ์นํ๋ฉ๋๋ค.
- ์ฌ๊ธฐ์ ๋ฌธ์ ์ ์ด ๋ณด์๋๋ค. **<span style="color:maroon"> Target Mean Encoding ๋ฐฉ์์ ์ฌ์ฉํ๋ฉด, ๊ฐ์ ์นดํ๊ณ ๋ฆฌ ๋ฐ์ดํฐ๋ผ๋, Train ๋ฐ์ดํฐ์ Test ๋ฐ์ดํฐ ๊ฐ์ TS๊ฐ ๋ฌ๋ผ์ง๋ ๊ฒ**์๋๋ค.
- ๊ทธ๋ ๋ค๋ฉด Ordered TS ๋ฐฉ์์ ๋ค๋ฅผ๊น์?


<br/>

### <span style="background-color:#fff5b1"><span style="color:maroon">2. Ordered Target Statistics
```python
def Random_Permutation(x):
    perm = np.random.permutation(len(x)) 
    x = x.iloc[perm].reset_index(drop=True) 
    return x

def Ordered_TS_for_binary_clf(X: np.ndarray, y: np.ndarray, a: float=0.1):
    x_list, y_list, ts_list = [], [], []
    
    for k, (x, y) in enumerate(zip(X, y)):
        if k == 0:
            ts_list.append(0)
        else:
            p = np.sum(y_list)/len(y_list)
            ts = (sum(np.array(y_list)[[i for i, x_ in enumerate(x_list) if x_ == x]]) + a*p) / (x_list.count(x) + a)
            ts_list.append(ts)
        
        x_list.append(x)
        y_list.append(y)
    
    return pd.DataFrame(ts_list)
```
- Binary Classification Task๋ฅผ ํ๊ณ ์ ํ  ๋, Ordered TS ๋ฐฉ์์ ๊ตฌํํ ๊ฒ์๋๋ค. 
- ๋จผ์  Random Permutation์ ์ํํด์ค ํ, Ordered TS๋ฅผ ๊ตฌํด์ค ๊ฒ์๋๋ค.

```python
permuted_train = Random_Permutation(df_train)
permuted_train['Ordered_TS_Embarked'] = Ordered_TS_for_binary_clf(permuted_train.Embarked, permuted_train.Survived, a=0.1)
permuted_train
```
```
    Embarked	Survived	Ordered_TS_Embarked
0	    S	        0	        0.000000
1	    S	        0	        0.000000
2	    S	        0	        0.000000
3	    S	        0	        0.000000
4	    S	        1	        0.000000
...	    ...	       ...	            ...
706	    S	        0	        0.350202
707	    Q	        0	        0.333434
708	    S	        0	        0.349522
709	    S	        1	        0.348844
710	    S	        0	        0.350104
```

```python
permuted_test = Random_Permutation(df_test)
permuted_test['Ordered_TS_Embarked'] = Ordered_TS_for_binary_clf(permuted_test.Embarked, permuted_test.Survived, a=0.1)
permuted_test
```
```
    Embarked	Survived	Ordered_TS_Embarked
0	    S	        0	        0.000000
1	    S	        0	        0.000000
2	    S	        1	        0.000000
3	    S	        1	        0.333333
4	    Q	        1	        0.500000
...	    ...	       ...	            ...
173	    S	        1	        0.284613
174	    S	        0	        0.290380
175	    C	        1	        0.551065
176	    Q	        1	        0.523050
177	    S	        0	        0.288063
```
- **<span style="color:maroon"> Ordered TS๋ ๊ฐ์ ์นดํ๊ณ ๋ฆฌ ๊ฐ์ด๋ผ ํ๋๋ผ๋ ๋์ผํ TS๋ก ์นํ๋์ง ์์ต๋๋ค.**
- ๋จผ์  **<span style="color:maroon"> ๋ฐ์ดํฐ์ artificial time์ ๋ถ์ฌํ๊ณ , TS๋ฅผ ๊ณ์ฐํ๋ ๋ฐ ์์ด์ ์ด์ ์ ๋ํ๋ ๋ฐ์ดํฐ์๋ง์ ์ด์ฉํ์ฌ ๊ณ์ฐํ๊ธฐ ๋๋ฌธ**์๋๋ค. ๋ฐ๋ผ์ Random Permutation์ ์ฌ๋ฌ ๋ฒ ์ํํด์ผ ํ๋ค๋ ํน์ง์ ๊ฐ์ง๋๋ค.
- ๊ทธ๋ฌ๋ <span style="color:maroon">  ํํ ๋ฆฌ์ผ์์๋ 1๋ฒ์ Permutation๋ง์ ์์๋ก ์ํํ์์ต๋๋ค.


### <span style="background-color:#fff5b1"><span style="color:maroon"> Expectation of TS
๊ทธ๋ ๋ค๋ฉด **<span style="color:maroon">Train ๋ฐ์ดํฐ์ TS์ ๊ธฐ๋๊ฐ๊ณผ, Test ๋ฐ์ดํฐ์ TS์ ๊ธฐ๋๊ฐ์ ์๋ก ๋์ผํ ๊น์?** ๋ ๋ฐฉ์์ด Property 1์ ๋ง์กฑํ๋์ง ๊ทธ ์ฌ๋ถ๋ฅผ ์์๋ณด๊ฒ ์ต๋๋ค.
```python
def ts_expectation_binary_clf(df_train: pd.DataFrame, df_test: pd.DataFrame, TS_col:str, y_col:str):
    exp1_train = np.mean(df_train[df_train[y_col] == 0][TS_col])
    exp2_train = np.mean(df_train[df_train[y_col] == 1][TS_col])
    
    exp1_test = np.mean(df_test[df_test[y_col] == 0][TS_col])
    exp2_test = np.mean(df_test[df_test[y_col] == 1][TS_col])
    
    print(f"Expectation of '{TS_col}' when y = 0: ", "[train]", np.round(exp1_train, 3), "[test]", np.round(exp1_test, 3))
    print(f"Expectation of '{TS_col}' when y = 1: ", "[train]", np.round(exp2_train, 3), "[test]", np.round(exp2_test, 3))
```
- ์๋ Binary Classification Task์ ์์ด TS์ ๊ธฐ๋๊ฐ์ ๊ตฌํ๋ ๋ฐฉ์์ ๊ตฌํํ ๊ฒ์๋๋ค.

๋จผ์  **Mean Target Encoding์ TS ๊ธฐ๋๊ฐ์ ํ์ธ**ํด๋ณด๊ฒ ์ต๋๋ค.
```python
ts_expectation_binary_clf(df_train_TS, df_test_TS, TS_col='TS', y_col='Survived')
```
```
Expectation of 'TS' when y = 0:  [train] 0.376 [test] 0.341
Expectation of 'TS' when y = 1:  [train] 0.404 [test] 0.407
```
- ํ์ธ ๊ฒฐ๊ณผ, train๊ณผ test ๊ฐ TS ๊ธฐ๋๊ฐ์ด ์ฝ๊ฐ์ ์ฐจ์ด๊ฐ ์์ผ๋ ๊ฑฐ์ ์ ์ฌํ ๊ฐ์ด ๋์ค๋ ๊ฒ์ ํ์ธํ์์ต๋๋ค.
- ๊ธฐ๋์๋ ๋ค๋ฅด๊ฒ, Mean TS๋ Property 1์ ์ด๋ ์ ๋ ๋ง์กฑํ๋ ๋ฏ ํฉ๋๋ค.

๋ค์์ผ๋ก๋ **Ordered TS์ ๊ธฐ๋๊ฐ์ ํ์ธ**ํค๋ณด๊ฒ ์ต๋๋ค.
```python
ts_expectation_binary_clf(permuted_train, permuted_test, TS_col='Ordered_TS_Embarked', y_col='Survived')
```
```
Expectation of 'Ordered_TS_Embarked' when y = 0:  [train] 0.369 [test] 0.348
Expectation of 'Ordered_TS_Embarked' when y = 1:  [train] 0.386 [test] 0.417
```
- ํ์ธ ๊ฒฐ๊ณผ, train๊ณผ test ๊ฐ TS ๊ธฐ๋๊ฐ์ด ์ฝ๊ฐ์ ์ฐจ์ด๊ฐ ์๊ณ , y=0์ผ ๋๋ ๊ธฐ๋๊ฐ์ด ๊ฑฐ์ ๋น์ทํ์ง๋ง, y=1์ผ ๋์๋ ๊ธฐ๋๊ฐ์ด ์กฐ๊ธ ๋ฌ๋ผ์ง๋๋ค.
- ๊ธฐ๋์๋ ๋ค๋ฅด๊ฒ, Ordered TS๋ Property 1์ ๋ง์กฑํ์ง ๋ชปํ๊ณ  ์์ต๋๋ค.


### <span style="background-color:#fff5b1"><span style="color:maroon"> Evaluate using single Decision Tree
๊ทธ๋ ๋ค๋ฉด ์์์ ๋ง๋  TS ๊ฐ์ ๊ฐ์ง๊ณ  ๊ฐ๊ฐ ์ฑ๋ฅ์ ๋์ถํด๋ณด๊ฒ ์ต๋๋ค. Train ๋ฐ์ดํฐ์ Test ๋ฐ์ดํฐ ๊ฐ **<span style="color:maroon"> TS Expectation ์ฐจ์ด๊ฐ ์ข์ Mean Target Encoding ๋ฐฉ์์ด Overfitting์ ๋ฐฉ์งํ์ฌ ๋ ์ข์ ์ฑ๋ฅ์ ๋ผ ์ ์์ ๊ฒ**์๋๋ค.   

```python
# Define Dataset
mean_TS_df = pd.concat([df_train_TS, df_test_TS])
ordered_TS_df = pd.concat([permuted_train, permuted_test])

# Define Single Decision Tree
tree_clf = DecisionTreeClassifier(random_state=2022)
```
- Dataset์ ์์ ๋ง๋ค์๋ ๊ฒ์ Train/Test๋ฅผ concatํ์ฌ ์ฌ์ฉํ๊ณ , ์ด๋ฅผ Cross Validation์ ํตํด ๊ฒ์ฆํ  ๊ฒ์๋๋ค.
- ๋ชจ๋ธ์ Single Decision tree๋ฅผ ํ์ฉํ๋ฉฐ, hyperparameter๋ default ๊ฐ์ ์ฌ์ฉํฉ๋๋ค.

๋จผ์  **Mean Target Encoding ๋ฐฉ์์์์ ์ฑ๋ฅ์ ํ์ธ**ํด๋ณด๊ฒ ์ต๋๋ค.
```python
mean_acc, mean_auroc = evaluate_model(tree_clf, np.reshape(mean_TS_df.TS.values, (-1, 1)), mean_TS_df.Survived)
```
```
Accuracy: 0.63 (0.04)
AUROC: 0.58 (0.05)
```
๋ค์์ผ๋ก๋ **Ordered Target Statistics ๋ฐฉ์์์์ ์ฑ๋ฅ์ ํ์ธ**ํด๋ณด๊ฒ ์ต๋๋ค.
```python
ordered_acc, ordered_auroc = evaluate_model(tree_clf, np.reshape(ordered_TS_df.Ordered_TS_Embarked.values, (-1, 1)), ordered_TS_df.Survived)
```
```
Accuracy: 0.54 (0.04)
AUROC: 0.51 (0.05)
```

<br/>

### Results
|                             | __Mean Target Encoding__| __Ordered Target Statistics__ |
|:---------------------------: |:-----------------------:|:-----------------------------:|
| __Diff in Expectation (y=0)__|  0.035           |   **0.021**   |
| __Diff in Expectation (y=1)__|  **0.003**           |   0.031   |
|      __Mean Accuracy (std)__| **0.63 (0.04)**       | 0.54 (0.04)  |
|         __Mean AUROC (std)__|   **0.58 (0.05)**        |  0.51 (0.05) |


<br/>

### <span style="background-color:#fff5b1">  [์คํ 9] ๊ฒฐ๊ณผ ํด์
- ๋ณธ ํํ ๋ฆฌ์ผ์์๋ Mean Target Encoding๊ณผ Ordered Target Statistics๋ฅผ ๊ตฌํํ์์ผ๋ฉฐ, ๊ฐ๊ฐ์ TS์ ๋ํ Expectation ๋ฐ ๊ธฐ๋ณธ์ ์ธ ์ฑ๋ฅ์ ๋์ถํ์์ต๋๋ค.   
- ์ด๋ **๊ธฐ<span style="color:maroon">๋์๋ ๋ฌ๋ฆฌ, Ordered TS์ TS ๊ธฐ๋๊ฐ์ด Property 1์ ๋ง์กฑํ์ง ๋ชปํจ๊ณผ ๋์์, Single Decision Tree๋ฅผ ํตํด ์ฑ๋ฅ์ ๋์ถํ ๊ฒฐ๊ณผ Accuracy์ AUROC ๊ด์ ์์ ๋ชจ๋ ๋ฎ์ ์ฑ๋ฅ์ ๊ธฐ๋ก**ํ์์ต๋๋ค.
- Property 1์ ๋ง์กฑํ์ง ๋ชปํ ๊ฒ์ Ordered TS์ Mean Target Encoding์ด ๋ชจ๋ ๋ง์ฐฌ๊ฐ์ง์ธ๋ฐ, ์ด๋ **<span style="color:maroon">๋ณธ ํํ ๋ฆฌ์ผ์์๋ Random Permutation์ ํ ๋ฒ๋ง ์ํํ์์ ๊ฐ์**ํด์ผ ํฉ๋๋ค. **<span style="color:maroon">์ค์  CatBoost ์๊ณ ๋ฆฌ์ฆ ๋ด์์๋ itertation ๋ง๋ค, tree๋ฅผ ์๋ก ์์ฑํ  ๋๋ง๋ค permutation์ ์ํํ๊ณ , ์ง์์ ์ธ Inference ํ๊ฒฝ์ ์กฐ์ฑ**ํฉ๋๋ค.
- ๋๋ถ์ด **<span style="color:maroon">๊ธฐ๋ณธ์ ์ธ ์ฑ๋ฅ์ด ๋ฎ์ ์ด์ ๋ Mean Target Encoding์ TS ๊ณ์ฐ ์ ํ์ต ๋ฐ์ดํฐ์ ๋ชจ๋  target ๊ฐ์ ํจ๊ป ํ์ฉํ๊ณ  ์๊ธฐ ๋๋ฌธ**์ผ ๊ฐ๋ฅ์ฑ์ด ๋์ต๋๋ค. ๋ณธ ํํ ๋ฆฌ์ผ์์๋ **<span style="color:maroon">๊ฒ์ฆ ์ ํด๋น ํ์ต ๋ฐ์ดํฐ์ ๊ฒ์ฆ ๋ฐ์ดํฐ๋ฅผ ๋ชจ๋ ํฉ์ณ Cross Validation์ ์ํ**ํ์๊ธฐ์ **<span style="color:maroon">Target ๊ฐ์ ์ด๋ฏธ ์ถฉ๋ถํ ํ์ฉํ Mean Target Encoding ๋ฐฉ์์ด ๋ ๋์ ์ฑ๋ฅ์ ๊ธฐ๋กํ  ์ ์์ผ๋ก ์ถ์ธก**๋ฉ๋๋ค.
- ์ฆ, ๋ณธ ํํ ๋ฆฌ์ผ์์ ํ ๋ฒ๋ง ์ํํ๋ random permutation์ ์ฌ๋ฌ ๋ฒ ์ํํ๊ณ , Ordered Boosting์ ํตํด Tree๋ฅผ ๋ง๋ค์ด๋๊ฐ๋ฉด, **<span style="color:maroon"><span style="background-color:#fff5b1">์ค์  CatBoost ์๊ณ ๋ฆฌ์ฆ์ Overfitting์ ๋ฐฉ์งํจ๊ณผ ๋์์ Categorical ๋ณ์๋ ์ ์ ํ ์ฒ๋ฆฌํ๋ ํ๋ฅญํ ๋ชจ๋ธ๋ก์ ํ์ฉํ  ์ ์์ ๊ฒ**์๋๋ค.

<br/>

-----
# Insights
์ง๊ธ๊น์ง๋ Bagging๊ณผ Boosting ์๊ณ ๋ฆฌ์ฆ์ ๊ธฐ๋ณธ๊ณผ ํจ๊ป, ๊ฐ ๊ธฐ๋ฒ์ ๋ํ ์๊ณ ๋ฆฌ์ฆ์ ๋์ ๋ฐฉ์ ๋ฐ ํน์ง์ ์คํ ๋ฐ Python code๋ฅผ ํตํด ์์๋ณด์์ต๋๋ค. ๊ทธ๋ ๋ค๋ฉด **Bagging๊ณผ Boosting ์ค ์ด๋ค ๊ฒ์ ์ฌ์ฉํด์ผ ํ ๊น์?**    

์ผ๋ฐ์ ์ผ๋ก **<span style="color:darkblue">Bagging์ ๋จ์ผ ๋ชจ๋ธ์ ์์ธก ์ฑ๋ฅ์ด ์ด๋ ์ ๋ ๋ณด์ฅ๋ ์ํฉ์์, Overfitting์ด ๋ฌธ์ ๊ฐ ๋๋ ๊ฒฝ์ฐ ์ด๋ฅผ ํด๊ฒฐํ๋ ๋ฐฉ๋ฒ๋ก ์ผ๋ก์ ์ฌ์ฉ**ํฉ๋๋ค. ๋ฐ๋ฉด **<span style="color:purple">Boosting์ Bias๋ฅผ ์ค์ด๋ฉฐ ์์ธก ์ฑ๋ฅ์ ๋์ด๊ณ ์ ํ๋ ๋ฐฉ๋ฒ๋ก ์ด๊ธฐ์, Bagging๋ณด๋ค๋ ์์ธก ์ฑ๋ฅ์ด ๋ ์ข์ต๋๋ค.**   

๊ทธ๋ฌ๋ ์ค์  ํ๋ก์ ํธ์์๋ ์๊ฐํ๊ฐ ๋ชฉ์ ์ด ์๋๋ผ๋ฉด, ๊ตณ์ด ๋จ์ผ ๋ชจ๋ธ์ ๋จผ์  ๋ง๋  ํ Ensemble Learning์ ์ํํ๋ ๊ฒฝ์ฐ๋ ๊ฑฐ์ ์๊ฒ ์ฃ . ๋ฐ๋ผ์ **<span style="background-color:#fff5b1">Bagging ๊ธฐ๋ฐ์ Random Forest๋ก ๋จผ์  ํ์ต์ ์งํํ์ฌ ํ์ฌ ๊ฐ์ง๊ณ  ์๋ Dataset์ผ๋ก ์ด๋ ์ ๋์ ์ฑ๋ฅ์ ๋ผ ์ ์์์ง ๊ฐ๋ ํด๋ณด๊ณ , ์ดํ ์ค์  ๋ชจ๋ธ์ Boosting ๊ณ์ด์ ๋ชจ๋ธ๋ก ์ต์ ํ ์์์ ๊ฑฐ์ณ ์ฑ๋ฅ์ ๋์ด์ฌ๋ฆฌ๋ ๊ฒ์ด ํจ์จ์ ์ธ ํ๋ก์ ํธ ์งํ ๋ฐฉ๋ฒ**์๋๋ค.  

ํนํ **CatBoost๋ Ordered Boosting์ ํตํด Overfitting ๋ฐฉ์ง์ ๋๋ถ์ด ์ฑ๋ฅ์ ๊ทน๋ํํจ๊ณผ ๋์์, Ordered TS๋ฅผ ํตํด Categorical ๋ณ์๋ฅผ ๋ค๋ฃจ๋ ๋ฐ์๋ ํ๋ฅญํ ๋ฐฉ๋ฒ๋ก **์๋๋ค.   

์ด๋ฌํ ์๊ณ ๋ฆฌ์ฆ์ ํน์ง์ ๊ฐ์ํ๊ณ , ๋ณธ ํํ ๋ฆฌ์ผ์ ํตํด ํ๋ก์ ํธ์์ ์ด๋ค Ensemble Model๋ฅผ ํํ์ฌ ํ๋ก์ ํธ๋ฅผ ์ํํ  ์ง ๋์์ด ๋์์ผ๋ฉด ํฉ๋๋ค. ๐ฒ

----
<br/>

# References
https://ysyblog.tistory.com/220
https://machinelearningmastery.com/bagging-ensemble-with-python/
https://machinelearningmastery.com/gradient-boosting-machine-ensemble-in-python/
https://www.kaggle.com/code/faressayah/xgboost-vs-lightgbm-vs-catboost-vs-adaboost
https://towardsdatascience.com/dealing-with-categorical-variables-by-using-target-encoder-a0f1733a4c69
https://m.blog.naver.com/baek2sm/221771893509
