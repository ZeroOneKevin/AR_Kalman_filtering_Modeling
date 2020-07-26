AR_Kalman_filtering Model

An autoregression model(AR model）could be written as

$$x_{_t} = \phi_{_{n1}} x_{_{t-1}}+\phi_{_{n2}} x_{_{t-2}}+...+\phi_{_{nn}} x_{_{t-n}}+\varepsilon_t$$

Transfer multiple AR models into a state space below：
$$
       \begin{pmatrix}
        x_{_1}^{_t+_1}  \\
        x_{_2}^{_t+_1}   \\
        \vdots\\
        x_{_n}^{_t+_1}   \\
        \end{pmatrix}
        = 
        \begin{pmatrix}
        \phi_{_{11}} ^{_t}& 0 & 0 & \cdots & 0 \\
        \phi_{_{21}}^{_t} & \phi_{_{22}}^{_t} & 0 & \cdots & 0 \\
        \vdots & \vdots & \vdots & \ddots & \vdots \\
         \phi_{_{n1}}^{_t} &  \phi_{_{n2}}^{_t} &  \phi_{_{n3}}^{_t} & \cdots &  \phi_{_{nn}}^{_t} \\
        \end{pmatrix} 
        \begin{pmatrix}
        x_{_1}^{_t}  \\
        x_{_2}^{_t}   \\
        \vdots\\
        x_{_n}^{_t}   \\
        \end{pmatrix}+
        \begin{pmatrix}
        \varepsilon_{_1}^{_t}  \\
        \varepsilon_{_2}^{_t}   \\
        \vdots\\
        \varepsilon_{_n}^{_t}   \\
        \end{pmatrix}w_t
$$

$$
        y^{_{t+1}}= x_{_n}^{_{t+1}}+v^{_{t+1}}
$$

Then we could use kalman filter to predict or filter a time sequence.
