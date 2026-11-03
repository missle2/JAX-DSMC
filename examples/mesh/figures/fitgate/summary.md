# Fitgate: adjoint vs central differences

Ratio = seed-mean adjoint / seed-mean FD at step h, ±2 s.e. of the adjoint mean; t = paired (adjoint - FD) mean / s.e.; spread = per-seed sd / |FD|.

## FD references

| gate | n | FD ±2 s.e. | FD(h/2)/FD(h) |
|---|---|---|---|
| sq_spec | 32 | ±1.5 % | 0.987 ± 0.031 |
| sq_diff | 32 | ±2.2 % | 1.003 ± 0.046 |
| c10_spec | 32 | ±2.5 % | 1.008 ± 0.048 |
| c10_spec_dt2 | 16 | ±5.7 % | 1.029 ± 0.093 |
| c10_spec_g2n | 16 | ±2.9 % | 0.999 ± 0.057 |
| c10_spec_g4n | 16 | ±3.3 % | 1.016 ± 0.069 |
| c10_spec_2dt | 16 | ±2.3 % | 0.996 ± 0.038 |
| c10_spec_h1000 | 8 | ±4.7 % | 1.067 ± 0.088 |
| c10_spec_mem | 8 | ±4.7 % | 1.067 ± 0.088 |
| c10_spec_gz001 | 8 | ±4.7 % | 1.067 ± 0.088 |
| c10_spec_two | 8 | ±4.7 % | 1.067 ± 0.088 |
| c10_spec_v816 | 8 | ±4.7 % | 1.067 ± 0.088 |
| c10_spec_kn2 | 16 | ±1.7 % | 1.004 ± 0.031 |
| c10_spec_kn10 | 16 | ±0.8 % | 0.996 ± 0.010 |
| c10_diff | 8 | ±3.3 % | 1.029 ± 0.139 |
| c10_diff_dt2 | 16 | ±3.5 % | 0.984 ± 0.081 |
| c10_diff_f2 | 16 | ±4.3 % | 0.966 ± 0.062 |
| c10_diff_g2n | 16 | ±2.7 % | 1.051 ± 0.070 |
| c10_diff_mem | 8 | ±3.3 % | 1.029 ± 0.139 |
| c10_diff_h600 | 8 | ±3.3 % | 1.029 ± 0.139 |
| c10_diff_kn2 | 16 | ±2.7 % | 0.994 ± 0.053 |
| c10_diff_kn10 | 16 | ±1.6 % | 1.013 ± 0.024 |
| af_drag_radius | 32 | ±8.8 % | 1.133 ± 0.160 |
| af_drag_mode2 | 32 | ±10.1 % | 0.979 ± 0.174 |

## sq_spec (squash M0.32 Kn0.5 specular)

| variant | base ratio | t | spread | pool1 ratio | t | spread |
|---|---|---|---|---|---|---|
| one | 0.968 ± 0.029 | -1.8 | 0.082 | 0.966 ± 0.032 | -1.9 | 0.092 |
| one_pool | 0.984 ± 0.005 | -1.9 | 0.015 | 0.968 ± 0.005 | -4.2 | 0.015 |
| along | 0.970 ± 0.017 | -2.4 | 0.048 | 0.968 ± 0.021 | -2.6 | 0.058 |
| along_pool | 0.985 ± 0.005 | -1.8 | 0.015 | 0.970 ± 0.005 | -3.9 | 0.014 |
| one_pool_plane | 1.001 ± 0.005 | +0.2 | 0.015 | 0.986 ± 0.005 | -1.8 | 0.015 |
| along_pool_plane | 1.000 ± 0.005 | +0.0 | 0.015 | 0.985 ± 0.005 | -2.0 | 0.014 |
| cell_plane | 0.968 ± 0.009 | -3.6 | 0.025 | 0.961 ± 0.013 | -4.0 | 0.036 |
| cell_along_plane | 0.977 ± 0.008 | -2.6 | 0.023 | 0.970 ± 0.009 | -3.4 | 0.025 |
| cell_pool_plane | 0.986 ± 0.005 | -1.7 | 0.015 | 0.972 ± 0.005 | -3.7 | 0.015 |
| cell_along_pool_plane | 0.992 ± 0.006 | -0.9 | 0.016 | 0.978 ± 0.005 | -2.8 | 0.015 |
| cellq_plane | 0.969 ± 0.009 | -3.5 | 0.024 | 0.966 ± 0.011 | -3.7 | 0.031 |
| cellq_along_plane | 0.978 ± 0.008 | -2.6 | 0.021 | 0.972 ± 0.007 | -3.5 | 0.019 |
| cellq_pool_plane | 0.983 ± 0.006 | -2.0 | 0.017 | 0.971 ± 0.005 | -3.7 | 0.015 |
| cellq_along_pool_plane | 0.989 ± 0.006 | -1.4 | 0.016 | 0.976 ± 0.005 | -3.0 | 0.016 |
| cellxv_plane | 0.964 ± 0.011 | -3.6 | 0.030 | 0.960 ± 0.009 | -4.2 | 0.026 |
| cellxv_along_plane | 0.972 ± 0.007 | -3.1 | 0.020 | 0.963 ± 0.006 | -4.3 | 0.018 |
| cellxv_pool_plane | 0.966 ± 0.006 | -4.1 | 0.017 | 0.955 ± 0.005 | -5.8 | 0.014 |
| cellxv_along_pool_plane | 0.969 ± 0.006 | -3.8 | 0.017 | 0.956 ± 0.005 | -5.6 | 0.015 |

## sq_diff (squash M0.32 Kn0.5 diffuse)

| variant | base ratio | t | spread | pool1 ratio | t | spread |
|---|---|---|---|---|---|---|
| one | 0.998 ± 0.141 | -0.0 | 0.398 | 0.963 ± 0.130 | -0.6 | 0.367 |
| one_pool | 0.914 ± 0.024 | -5.1 | 0.067 | 0.882 ± 0.023 | -6.9 | 0.066 |
| along | 1.003 ± 0.059 | +0.1 | 0.166 | 0.944 ± 0.059 | -1.7 | 0.168 |
| along_pool | 0.971 ± 0.023 | -1.7 | 0.066 | 0.942 ± 0.022 | -3.5 | 0.062 |
| one_pool_plane | 0.927 ± 0.024 | -4.2 | 0.069 | 0.894 ± 0.023 | -6.1 | 0.066 |
| along_pool_plane | 0.982 ± 0.024 | -1.1 | 0.067 | 0.953 ± 0.022 | -2.8 | 0.063 |
| cell_plane | 0.905 ± 0.052 | -3.4 | 0.146 | 0.849 ± 0.053 | -5.1 | 0.149 |
| cell_along_plane | 1.018 ± 0.030 | +0.9 | 0.085 | 0.981 ± 0.027 | -1.0 | 0.077 |
| cell_pool_plane | 0.870 ± 0.023 | -7.7 | 0.065 | 0.841 ± 0.023 | -9.3 | 0.065 |
| cell_along_pool_plane | 1.005 ± 0.024 | +0.3 | 0.068 | 0.974 ± 0.023 | -1.5 | 0.064 |
| cellq_plane | 0.909 ± 0.050 | -3.5 | 0.142 | 0.847 ± 0.051 | -5.5 | 0.144 |
| cellq_along_plane | 1.018 ± 0.029 | +0.9 | 0.083 | 0.982 ± 0.027 | -0.9 | 0.076 |
| cellq_pool_plane | 0.877 ± 0.023 | -7.2 | 0.066 | 0.849 ± 0.023 | -8.8 | 0.065 |
| cellq_along_pool_plane | 1.007 ± 0.024 | +0.4 | 0.069 | 0.977 ± 0.023 | -1.4 | 0.064 |
| cellxv_plane | 0.941 ± 0.043 | -2.6 | 0.121 | 0.870 ± 0.047 | -5.1 | 0.132 |
| cellxv_along_plane | 1.014 ± 0.026 | +0.8 | 0.073 | 0.968 ± 0.025 | -1.8 | 0.071 |
| cellxv_pool_plane | 0.910 ± 0.025 | -5.1 | 0.070 | 0.876 ± 0.026 | -6.9 | 0.074 |
| cellxv_along_pool_plane | 0.979 ± 0.024 | -1.2 | 0.067 | 0.948 ± 0.022 | -3.1 | 0.063 |

## c10_spec (circle M10 Kn0.5 specular)

| variant | base ratio | t | spread | pool1 ratio | t | spread |
|---|---|---|---|---|---|---|
| one | 1.134 ± 0.021 | +8.0 | 0.060 | 1.122 ± 0.022 | +7.6 | 0.063 |
| one_pool | 1.141 ± 0.007 | +11.2 | 0.019 | 1.125 ± 0.007 | +9.7 | 0.020 |
| along | 1.126 ± 0.007 | +9.7 | 0.019 | 1.121 ± 0.007 | +9.0 | 0.021 |
| along_pool | 1.110 ± 0.005 | +8.9 | 0.013 | 1.108 ± 0.006 | +8.3 | 0.017 |
| one_pool_plane | 1.142 ± 0.007 | +11.2 | 0.019 | 1.125 ± 0.007 | +9.7 | 0.020 |
| along_pool_plane | 1.111 ± 0.005 | +9.0 | 0.013 | 1.108 ± 0.006 | +8.3 | 0.017 |
| cell_plane | 1.143 ± 0.028 | +7.4 | 0.079 | 1.124 ± 0.031 | +6.8 | 0.086 |
| cell_along_plane | 1.128 ± 0.008 | +9.8 | 0.022 | 1.122 ± 0.009 | +9.0 | 0.026 |
| cell_pool_plane | 1.204 ± 0.008 | +15.7 | 0.022 | 1.179 ± 0.009 | +13.4 | 0.025 |
| cell_along_pool_plane | 1.141 ± 0.005 | +11.2 | 0.014 | 1.136 ± 0.006 | +10.3 | 0.017 |
| cellq_plane | 1.138 ± 0.031 | +6.6 | 0.088 | 1.122 ± 0.032 | +6.0 | 0.092 |
| cellq_along_plane | 1.134 ± 0.009 | +10.1 | 0.025 | 1.130 ± 0.010 | +9.4 | 0.028 |
| cellq_pool_plane | 1.200 ± 0.008 | +15.6 | 0.022 | 1.175 ± 0.008 | +13.1 | 0.022 |
| cellq_along_pool_plane | 1.146 ± 0.005 | +11.6 | 0.014 | 1.142 ± 0.006 | +10.6 | 0.017 |
| cellxv_plane | 1.128 ± 0.033 | +5.7 | 0.094 | 1.100 ± 0.032 | +4.6 | 0.089 |
| cellxv_along_plane | 1.112 ± 0.009 | +8.8 | 0.026 | 1.106 ± 0.010 | +8.0 | 0.027 |
| cellxv_pool_plane | 1.085 ± 0.007 | +6.4 | 0.020 | 1.073 ± 0.008 | +5.4 | 0.023 |
| cellxv_along_pool_plane | 1.090 ± 0.005 | +7.1 | 0.014 | 1.086 ± 0.006 | +6.5 | 0.017 |

## c10_spec_dt2 (circle M10 specular, dt/2)

| variant | base ratio | t | spread |
|---|---|---|---|
| one | 1.154 ± 0.036 | +5.0 | 0.073 |
| one_pool | 1.162 ± 0.021 | +5.5 | 0.042 |
| along | 1.128 ± 0.021 | +4.5 | 0.042 |
| along_pool | 1.120 ± 0.020 | +4.3 | 0.041 |
| one_pool_plane | 1.163 ± 0.021 | +5.5 | 0.042 |
| along_pool_plane | 1.120 ± 0.020 | +4.3 | 0.040 |
| cell_plane | 1.165 ± 0.065 | +3.7 | 0.130 |
| cell_along_plane | 1.138 ± 0.024 | +4.6 | 0.048 |
| cell_pool_plane | 1.179 ± 0.022 | +6.0 | 0.044 |
| cell_along_pool_plane | 1.143 ± 0.020 | +5.1 | 0.040 |
| cellq_plane | 1.159 ± 0.065 | +3.6 | 0.130 |
| cellq_along_plane | 1.140 ± 0.024 | +4.7 | 0.047 |
| cellq_pool_plane | 1.163 ± 0.022 | +5.6 | 0.045 |
| cellq_along_pool_plane | 1.145 ± 0.020 | +5.3 | 0.039 |
| cellxv_plane | 1.133 ± 0.069 | +2.7 | 0.138 |
| cellxv_along_plane | 1.128 ± 0.022 | +4.3 | 0.043 |
| cellxv_pool_plane | 1.117 ± 0.021 | +4.2 | 0.043 |
| cellxv_along_pool_plane | 1.104 ± 0.020 | +3.8 | 0.040 |

## c10_spec_g2n (circle M10 specular, cells/2)

| variant | base ratio | t | spread |
|---|---|---|---|
| one | 1.092 ± 0.171 | +1.1 | 0.343 |
| one_pool | 1.139 ± 0.014 | +8.3 | 0.027 |
| along | 1.087 ± 0.027 | +4.0 | 0.054 |
| along_pool | 1.053 ± 0.009 | +4.0 | 0.019 |
| one_pool_plane | 1.139 ± 0.013 | +8.4 | 0.026 |
| along_pool_plane | 1.053 ± 0.009 | +4.0 | 0.019 |
| cell_plane | 1.130 ± 0.097 | +2.6 | 0.194 |
| cell_along_plane | 1.087 ± 0.025 | +4.7 | 0.051 |
| cell_pool_plane | 1.095 ± 0.024 | +5.5 | 0.048 |
| cell_along_pool_plane | 1.058 ± 0.012 | +4.2 | 0.023 |
| cellq_plane | 1.126 ± 0.088 | +2.7 | 0.176 |
| cellq_along_plane | 1.087 ± 0.022 | +4.9 | 0.043 |
| cellq_pool_plane | 1.080 ± 0.026 | +4.6 | 0.052 |
| cellq_along_pool_plane | 1.064 ± 0.012 | +4.5 | 0.023 |
| cellxv_plane | 1.133 ± 0.101 | +2.6 | 0.202 |
| cellxv_along_plane | 1.085 ± 0.023 | +5.0 | 0.046 |
| cellxv_pool_plane | 0.993 ± 0.034 | -0.4 | 0.067 |
| cellxv_along_pool_plane | 1.011 ± 0.012 | +0.9 | 0.025 |

## c10_spec_g4n (circle M10 specular, cells/4)

| variant | base ratio | t | spread |
|---|---|---|---|
| one | nan ± nan | +nan | nan |
| one_pool | nan ± nan | +nan | nan |
| along | nan ± nan | +nan | nan |
| along_pool | nan ± nan | +nan | nan |
| one_pool_plane | 1.103 ± 0.108 | +1.9 | 0.216 |
| along_pool_plane | 1.110 ± 0.019 | +5.6 | 0.039 |
| cell_plane | nan ± nan | +nan | nan |
| cell_along_plane | nan ± nan | +nan | nan |
| cell_pool_plane | 1.095 ± 0.102 | +1.8 | 0.205 |
| cell_along_pool_plane | 1.107 ± 0.028 | +4.4 | 0.056 |
| cellq_plane | nan ± nan | +nan | nan |
| cellq_along_plane | nan ± nan | +nan | nan |
| cellq_pool_plane | nan ± nan | +nan | nan |
| cellq_along_pool_plane | nan ± nan | +nan | nan |
| cellxv_plane | nan ± nan | +nan | nan |
| cellxv_along_plane | nan ± nan | +nan | nan |
| cellxv_pool_plane | nan ± nan | +nan | nan |
| cellxv_along_pool_plane | 1.079 ± 0.022 | +4.1 | 0.045 |

## c10_spec_2dt (circle M10 specular, 2 dt)

| variant | base ratio | t | spread |
|---|---|---|---|
| one | nan ± nan | +nan | nan |
| one_pool | nan ± nan | +nan | nan |
| along | nan ± nan | +nan | nan |
| along_pool | nan ± nan | +nan | nan |
| one_pool_plane | 1.076 ± 0.005 | +6.8 | 0.009 |
| along_pool_plane | 1.044 ± 0.004 | +3.8 | 0.008 |
| cell_plane | nan ± nan | +nan | nan |
| cell_along_plane | nan ± nan | +nan | nan |
| cell_pool_plane | 1.150 ± 0.009 | +14.4 | 0.017 |
| cell_along_pool_plane | 1.078 ± 0.004 | +6.6 | 0.009 |
| cellq_plane | nan ± nan | +nan | nan |
| cellq_along_plane | nan ± nan | +nan | nan |
| cellq_pool_plane | nan ± nan | +nan | nan |
| cellq_along_pool_plane | nan ± nan | +nan | nan |
| cellxv_plane | nan ± nan | +nan | nan |
| cellxv_along_plane | nan ± nan | +nan | nan |
| cellxv_pool_plane | nan ± nan | +nan | nan |
| cellxv_along_pool_plane | 1.029 ± 0.005 | +2.5 | 0.010 |

## c10_spec_h1000 (circle M10 specular, 1000 captured steps)

| variant | base ratio | t | spread |
|---|---|---|---|
| one | 1.077 ± 0.065 | +1.8 | 0.092 |
| one_pool | 1.103 ± 0.015 | +3.8 | 0.021 |
| along | 1.088 ± 0.014 | +3.6 | 0.020 |
| along_pool | 1.067 ± 0.011 | +3.3 | 0.016 |
| one_pool_plane | 1.103 ± 0.015 | +3.8 | 0.022 |
| along_pool_plane | 1.067 ± 0.011 | +3.3 | 0.016 |
| cell_plane | 1.136 ± 0.098 | +2.0 | 0.139 |
| cell_along_plane | 1.097 ± 0.020 | +3.2 | 0.028 |
| cell_pool_plane | 1.181 ± 0.019 | +6.6 | 0.027 |
| cell_along_pool_plane | 1.099 ± 0.011 | +4.5 | 0.015 |
| cellq_plane | 1.127 ± 0.103 | +1.8 | 0.145 |
| cellq_along_plane | 1.102 ± 0.014 | +3.6 | 0.020 |
| cellq_pool_plane | 1.182 ± 0.019 | +7.3 | 0.027 |
| cellq_along_pool_plane | 1.107 ± 0.012 | +5.1 | 0.018 |
| cellxv_plane | 1.157 ± 0.131 | +1.8 | 0.186 |
| cellxv_along_plane | 1.088 ± 0.018 | +3.0 | 0.025 |
| cellxv_pool_plane | 1.034 ± 0.027 | +1.3 | 0.038 |
| cellxv_along_pool_plane | 1.050 ± 0.013 | +2.6 | 0.019 |

## c10_spec_mem (circle M10 specular (+ membership, not in total))

| variant | base ratio | t | spread |
|---|---|---|---|
| one | nan ± nan | +nan | nan |
| one_pool | nan ± nan | +nan | nan |
| along | nan ± nan | +nan | nan |
| along_pool | nan ± nan | +nan | nan |
| one_pool_plane | 1.133 ± 0.011 | +5.7 | 0.015 |
| along_pool_plane | 1.101 ± 0.006 | +4.4 | 0.009 |
| cell_plane | nan ± nan | +nan | nan |
| cell_along_plane | nan ± nan | +nan | nan |
| cell_pool_plane | 1.186 ± 0.014 | +7.2 | 0.019 |
| cell_along_pool_plane | 1.131 ± 0.007 | +5.4 | 0.010 |
| cellq_plane | nan ± nan | +nan | nan |
| cellq_along_plane | nan ± nan | +nan | nan |
| cellq_pool_plane | nan ± nan | +nan | nan |
| cellq_along_pool_plane | nan ± nan | +nan | nan |
| cellxv_plane | nan ± nan | +nan | nan |
| cellxv_along_plane | nan ± nan | +nan | nan |
| cellxv_pool_plane | nan ± nan | +nan | nan |
| cellxv_along_pool_plane | 1.080 ± 0.007 | +3.4 | 0.010 |

## c10_spec_gz001 (circle M10 specular, grazing 0.001)

| variant | base ratio | t | spread |
|---|---|---|---|
| one | nan ± nan | +nan | nan |
| one_pool | nan ± nan | +nan | nan |
| along | nan ± nan | +nan | nan |
| along_pool | nan ± nan | +nan | nan |
| one_pool_plane | 1.133 ± 0.011 | +5.8 | 0.015 |
| along_pool_plane | 1.102 ± 0.006 | +4.5 | 0.009 |
| cell_plane | nan ± nan | +nan | nan |
| cell_along_plane | nan ± nan | +nan | nan |
| cell_pool_plane | 1.187 ± 0.013 | +7.2 | 0.019 |
| cell_along_pool_plane | 1.131 ± 0.007 | +5.4 | 0.010 |
| cellq_plane | nan ± nan | +nan | nan |
| cellq_along_plane | nan ± nan | +nan | nan |
| cellq_pool_plane | nan ± nan | +nan | nan |
| cellq_along_pool_plane | nan ± nan | +nan | nan |
| cellxv_plane | nan ± nan | +nan | nan |
| cellxv_along_plane | nan ± nan | +nan | nan |
| cellxv_pool_plane | nan ± nan | +nan | nan |
| cellxv_along_pool_plane | 1.081 ± 0.007 | +3.4 | 0.010 |

## c10_spec_two (circle M10 specular, two-sided fits)

| variant | base ratio | t | spread |
|---|---|---|---|
| one | nan ± nan | +nan | nan |
| one_pool | nan ± nan | +nan | nan |
| along | nan ± nan | +nan | nan |
| along_pool | nan ± nan | +nan | nan |
| one_pool_plane | 1.119 ± 0.009 | +5.0 | 0.012 |
| along_pool_plane | 1.097 ± 0.004 | +4.1 | 0.006 |
| cell_plane | nan ± nan | +nan | nan |
| cell_along_plane | nan ± nan | +nan | nan |
| cell_pool_plane | 1.169 ± 0.015 | +6.6 | 0.021 |
| cell_along_pool_plane | 1.126 ± 0.007 | +5.0 | 0.010 |
| cellq_plane | nan ± nan | +nan | nan |
| cellq_along_plane | nan ± nan | +nan | nan |
| cellq_pool_plane | nan ± nan | +nan | nan |
| cellq_along_pool_plane | nan ± nan | +nan | nan |
| cellxv_plane | nan ± nan | +nan | nan |
| cellxv_along_plane | nan ± nan | +nan | nan |
| cellxv_pool_plane | nan ± nan | +nan | nan |
| cellxv_along_pool_plane | 1.076 ± 0.006 | +3.1 | 0.009 |

## c10_spec_v816 (circle M10 specular, virtual 8x16)

| variant | base ratio | t | spread |
|---|---|---|---|
| one | nan ± nan | +nan | nan |
| one_pool | nan ± nan | +nan | nan |
| along | nan ± nan | +nan | nan |
| along_pool | nan ± nan | +nan | nan |
| one_pool_plane | 1.128 ± 0.011 | +5.5 | 0.016 |
| along_pool_plane | 1.097 ± 0.008 | +4.2 | 0.011 |
| cell_plane | nan ± nan | +nan | nan |
| cell_along_plane | nan ± nan | +nan | nan |
| cell_pool_plane | 1.181 ± 0.013 | +6.9 | 0.019 |
| cell_along_pool_plane | 1.126 ± 0.008 | +5.1 | 0.011 |
| cellq_plane | nan ± nan | +nan | nan |
| cellq_along_plane | nan ± nan | +nan | nan |
| cellq_pool_plane | nan ± nan | +nan | nan |
| cellq_along_pool_plane | nan ± nan | +nan | nan |
| cellxv_plane | nan ± nan | +nan | nan |
| cellxv_along_plane | nan ± nan | +nan | nan |
| cellxv_pool_plane | nan ± nan | +nan | nan |
| cellxv_along_pool_plane | 1.075 ± 0.007 | +3.2 | 0.011 |

## c10_spec_kn2 (circle M10 Kn2 specular)

| variant | base ratio | t | spread |
|---|---|---|---|
| one | 1.017 ± 0.012 | +1.5 | 0.024 |
| one_pool | 1.023 ± 0.006 | +2.9 | 0.013 |
| along | 1.019 ± 0.008 | +2.2 | 0.015 |
| along_pool | 1.015 ± 0.006 | +1.8 | 0.012 |
| one_pool_plane | 1.024 ± 0.007 | +2.9 | 0.013 |
| along_pool_plane | 1.015 ± 0.006 | +1.8 | 0.012 |
| cell_plane | 1.038 ± 0.014 | +2.8 | 0.029 |
| cell_along_plane | 1.024 ± 0.007 | +2.6 | 0.015 |
| cell_pool_plane | 1.055 ± 0.007 | +7.0 | 0.014 |
| cell_along_pool_plane | 1.021 ± 0.006 | +2.4 | 0.012 |
| cellq_plane | 1.029 ± 0.015 | +2.0 | 0.030 |
| cellq_along_plane | 1.022 ± 0.006 | +2.5 | 0.012 |
| cellq_pool_plane | 1.078 ± 0.007 | +10.5 | 0.014 |
| cellq_along_pool_plane | 1.023 ± 0.006 | +2.8 | 0.012 |
| cellxv_plane | 1.048 ± 0.020 | +3.1 | 0.040 |
| cellxv_along_plane | 1.021 ± 0.007 | +2.3 | 0.014 |
| cellxv_pool_plane | 1.014 ± 0.007 | +1.8 | 0.015 |
| cellxv_along_pool_plane | 1.013 ± 0.006 | +1.5 | 0.011 |

## c10_spec_kn10 (circle M10 Kn10 specular)

| variant | base ratio | t | spread |
|---|---|---|---|
| one | 1.012 ± 0.006 | +2.8 | 0.012 |
| one_pool | 1.006 ± 0.004 | +1.8 | 0.009 |
| along | 1.008 ± 0.005 | +2.2 | 0.010 |
| along_pool | 1.004 ± 0.004 | +1.1 | 0.009 |
| one_pool_plane | 1.006 ± 0.004 | +1.8 | 0.009 |
| along_pool_plane | 1.004 ± 0.004 | +1.1 | 0.009 |
| cell_plane | 1.013 ± 0.006 | +3.3 | 0.012 |
| cell_along_plane | 1.008 ± 0.005 | +2.4 | 0.009 |
| cell_pool_plane | 1.011 ± 0.004 | +3.1 | 0.008 |
| cell_along_pool_plane | 1.006 ± 0.004 | +1.7 | 0.008 |
| cellq_plane | 1.013 ± 0.006 | +3.3 | 0.012 |
| cellq_along_plane | 1.008 ± 0.005 | +2.4 | 0.009 |
| cellq_pool_plane | 1.016 ± 0.004 | +4.7 | 0.007 |
| cellq_along_pool_plane | 1.006 ± 0.004 | +1.7 | 0.009 |
| cellxv_plane | 1.015 ± 0.006 | +3.9 | 0.012 |
| cellxv_along_plane | 1.007 ± 0.004 | +2.1 | 0.009 |
| cellxv_pool_plane | 1.006 ± 0.004 | +1.7 | 0.008 |
| cellxv_along_pool_plane | 1.005 ± 0.004 | +1.3 | 0.008 |

## c10_diff (circle M10 Kn0.5 diffuse)

| variant | base ratio | t | spread |
|---|---|---|---|
| one | 0.836 ± 0.121 | -3.2 | 0.172 |
| one_pool | 0.923 ± 0.038 | -3.7 | 0.054 |
| along | 0.787 ± 0.222 | -2.2 | 0.314 |
| along_pool | 0.874 ± 0.052 | -5.4 | 0.074 |
| one_pool_plane | 0.927 ± 0.038 | -3.5 | 0.054 |
| along_pool_plane | 0.878 ± 0.052 | -5.2 | 0.074 |
| cell_plane | 0.903 ± 0.089 | -2.0 | 0.125 |
| cell_along_plane | 0.887 ± 0.051 | -4.4 | 0.072 |
| cell_pool_plane | 0.989 ± 0.043 | -0.5 | 0.060 |
| cell_along_pool_plane | 0.949 ± 0.054 | -2.1 | 0.076 |
| cellq_plane | 0.907 ± 0.098 | -1.7 | 0.139 |
| cellq_along_plane | 0.892 ± 0.061 | -3.4 | 0.087 |
| cellq_pool_plane | 0.974 ± 0.043 | -1.1 | 0.060 |
| cellq_along_pool_plane | 0.945 ± 0.054 | -2.2 | 0.076 |
| cellxv_plane | 0.961 ± 0.202 | -0.4 | 0.286 |
| cellxv_along_plane | 0.873 ± 0.098 | -2.9 | 0.139 |
| cellxv_pool_plane | 0.845 ± 0.042 | -8.8 | 0.059 |
| cellxv_along_pool_plane | 0.857 ± 0.053 | -6.1 | 0.075 |

## c10_diff_dt2 (circle M10 diffuse, dt/2)

| variant | base ratio | t | spread |
|---|---|---|---|
| one | 0.765 ± 0.165 | -2.9 | 0.331 |
| one_pool | 0.901 ± 0.050 | -3.4 | 0.100 |
| along | 0.727 ± 0.121 | -4.1 | 0.242 |
| along_pool | 0.781 ± 0.086 | -4.4 | 0.172 |
| one_pool_plane | 0.906 ± 0.050 | -3.2 | 0.100 |
| along_pool_plane | 0.785 ± 0.086 | -4.3 | 0.172 |
| cell_plane | 0.697 ± 0.100 | -6.1 | 0.201 |
| cell_along_plane | 0.760 ± 0.092 | -4.5 | 0.184 |
| cell_pool_plane | 0.965 ± 0.047 | -1.2 | 0.094 |
| cell_along_pool_plane | 0.860 ± 0.087 | -2.8 | 0.175 |
| cellq_plane | 0.703 ± 0.100 | -6.1 | 0.201 |
| cellq_along_plane | 0.765 ± 0.096 | -4.3 | 0.191 |
| cellq_pool_plane | 0.971 ± 0.048 | -1.0 | 0.095 |
| cellq_along_pool_plane | 0.865 ± 0.088 | -2.6 | 0.176 |
| cellxv_plane | 0.572 ± 0.193 | -4.2 | 0.385 |
| cellxv_along_plane | 0.675 ± 0.101 | -5.4 | 0.203 |
| cellxv_pool_plane | 0.862 ± 0.044 | -5.0 | 0.087 |
| cellxv_along_pool_plane | 0.779 ± 0.087 | -4.4 | 0.174 |

## c10_diff_f2 (circle M10 diffuse, N/2)

| variant | base ratio | t | spread |
|---|---|---|---|
| one | 0.778 ± 0.132 | -3.5 | 0.264 |
| one_pool | 0.895 ± 0.066 | -2.8 | 0.132 |
| along | 0.829 ± 0.130 | -2.8 | 0.260 |
| along_pool | 0.863 ± 0.096 | -2.7 | 0.193 |
| one_pool_plane | 0.899 ± 0.066 | -2.7 | 0.132 |
| along_pool_plane | 0.867 ± 0.096 | -2.6 | 0.193 |
| cell_plane | 0.821 ± 0.138 | -2.7 | 0.276 |
| cell_along_plane | 0.883 ± 0.108 | -2.2 | 0.216 |
| cell_pool_plane | 0.928 ± 0.063 | -2.1 | 0.127 |
| cell_along_pool_plane | 0.917 ± 0.097 | -1.6 | 0.195 |
| cellq_plane | 0.841 ± 0.126 | -2.7 | 0.251 |
| cellq_along_plane | 0.893 ± 0.103 | -2.1 | 0.207 |
| cellq_pool_plane | 0.912 ± 0.063 | -2.5 | 0.126 |
| cellq_along_pool_plane | 0.914 ± 0.097 | -1.7 | 0.195 |
| cellxv_plane | 0.689 ± 0.289 | -2.2 | 0.579 |
| cellxv_along_plane | 0.839 ± 0.116 | -2.7 | 0.232 |
| cellxv_pool_plane | 0.784 ± 0.064 | -6.0 | 0.128 |
| cellxv_along_pool_plane | 0.827 ± 0.097 | -3.3 | 0.195 |

## c10_diff_g2n (circle M10 diffuse, cells/2)

| variant | base ratio | t | spread |
|---|---|---|---|
| one | nan ± nan | +nan | nan |
| one_pool | nan ± nan | +nan | nan |
| along | nan ± nan | +nan | nan |
| along_pool | nan ± nan | +nan | nan |
| one_pool_plane | 1.431 ± 0.116 | +7.5 | 0.232 |
| along_pool_plane | 0.981 ± 0.136 | -0.3 | 0.271 |
| cell_plane | nan ± nan | +nan | nan |
| cell_along_plane | nan ± nan | +nan | nan |
| cell_pool_plane | 1.402 ± 0.106 | +7.6 | 0.213 |
| cell_along_pool_plane | 1.015 ± 0.134 | +0.2 | 0.269 |
| cellq_plane | nan ± nan | +nan | nan |
| cellq_along_plane | nan ± nan | +nan | nan |
| cellq_pool_plane | nan ± nan | +nan | nan |
| cellq_along_pool_plane | nan ± nan | +nan | nan |
| cellxv_plane | nan ± nan | +nan | nan |
| cellxv_along_plane | nan ± nan | +nan | nan |
| cellxv_pool_plane | nan ± nan | +nan | nan |
| cellxv_along_pool_plane | 0.930 ± 0.135 | -1.1 | 0.270 |

## c10_diff_mem (circle M10 diffuse (+ membership, not in total))

| variant | base ratio | t | spread |
|---|---|---|---|
| one | nan ± nan | +nan | nan |
| one_pool | nan ± nan | +nan | nan |
| along | nan ± nan | +nan | nan |
| along_pool | nan ± nan | +nan | nan |
| one_pool_plane | 0.927 ± 0.038 | -3.5 | 0.054 |
| along_pool_plane | 0.878 ± 0.052 | -5.2 | 0.074 |
| cell_plane | nan ± nan | +nan | nan |
| cell_along_plane | nan ± nan | +nan | nan |
| cell_pool_plane | 0.989 ± 0.043 | -0.5 | 0.060 |
| cell_along_pool_plane | 0.949 ± 0.054 | -2.1 | 0.076 |
| cellq_plane | nan ± nan | +nan | nan |
| cellq_along_plane | nan ± nan | +nan | nan |
| cellq_pool_plane | nan ± nan | +nan | nan |
| cellq_along_pool_plane | nan ± nan | +nan | nan |
| cellxv_plane | nan ± nan | +nan | nan |
| cellxv_along_plane | nan ± nan | +nan | nan |
| cellxv_pool_plane | nan ± nan | +nan | nan |
| cellxv_along_pool_plane | 0.857 ± 0.053 | -6.1 | 0.075 |

## c10_diff_h600 (circle M10 diffuse, 600 captured steps)

| variant | base ratio | t | spread |
|---|---|---|---|
| one | 0.909 ± 0.242 | -0.8 | 0.342 |
| one_pool | 0.907 ± 0.041 | -3.5 | 0.058 |
| along | 0.875 ± 0.318 | -0.9 | 0.449 |
| along_pool | 0.931 ± 0.111 | -1.3 | 0.157 |
| one_pool_plane | 0.912 ± 0.041 | -3.3 | 0.058 |
| along_pool_plane | 0.936 ± 0.112 | -1.2 | 0.158 |
| cell_plane | 0.972 ± 0.140 | -0.4 | 0.197 |
| cell_along_plane | 0.979 ± 0.125 | -0.4 | 0.176 |
| cell_pool_plane | 0.931 ± 0.036 | -3.1 | 0.051 |
| cell_along_pool_plane | 1.011 ± 0.112 | +0.2 | 0.159 |
| cellq_plane | 0.961 ± 0.162 | -0.5 | 0.230 |
| cellq_along_plane | 0.983 ± 0.140 | -0.3 | 0.198 |
| cellq_pool_plane | 0.912 ± 0.037 | -4.0 | 0.052 |
| cellq_along_pool_plane | 1.003 ± 0.113 | +0.1 | 0.159 |
| cellxv_plane | 0.999 ± 0.341 | -0.0 | 0.482 |
| cellxv_along_plane | 0.961 ± 0.161 | -0.5 | 0.228 |
| cellxv_pool_plane | 0.830 ± 0.038 | -8.7 | 0.054 |
| cellxv_along_pool_plane | 0.922 ± 0.114 | -1.5 | 0.161 |

## c10_diff_kn2 (circle M10 Kn2 diffuse)

| variant | base ratio | t | spread |
|---|---|---|---|
| one | 0.885 ± 0.069 | -3.0 | 0.137 |
| one_pool | 0.930 ± 0.057 | -2.0 | 0.114 |
| along | 0.897 ± 0.051 | -3.6 | 0.102 |
| along_pool | 0.909 ± 0.055 | -2.7 | 0.110 |
| one_pool_plane | 0.934 ± 0.057 | -1.9 | 0.114 |
| along_pool_plane | 0.913 ± 0.055 | -2.6 | 0.110 |
| cell_plane | 0.865 ± 0.067 | -3.5 | 0.133 |
| cell_along_plane | 0.904 ± 0.052 | -3.1 | 0.103 |
| cell_pool_plane | 1.003 ± 0.060 | +0.1 | 0.121 |
| cell_along_pool_plane | 0.988 ± 0.058 | -0.3 | 0.117 |
| cellq_plane | 0.867 ± 0.071 | -3.2 | 0.143 |
| cellq_along_plane | 0.909 ± 0.053 | -2.9 | 0.106 |
| cellq_pool_plane | 1.007 ± 0.061 | +0.2 | 0.121 |
| cellq_along_pool_plane | 0.992 ± 0.059 | -0.2 | 0.117 |
| cellxv_plane | 0.846 ± 0.075 | -3.6 | 0.150 |
| cellxv_along_plane | 0.883 ± 0.056 | -3.5 | 0.112 |
| cellxv_pool_plane | 0.941 ± 0.058 | -1.7 | 0.116 |
| cellxv_along_pool_plane | 0.940 ± 0.056 | -1.8 | 0.113 |

## c10_diff_kn10 (circle M10 Kn10 diffuse)

| variant | base ratio | t | spread |
|---|---|---|---|
| one | 0.856 ± 0.025 | -12.9 | 0.050 |
| one_pool | 0.930 ± 0.017 | -6.8 | 0.034 |
| along | 0.869 ± 0.024 | -10.7 | 0.049 |
| along_pool | 0.907 ± 0.018 | -8.3 | 0.037 |
| one_pool_plane | 0.933 ± 0.017 | -6.5 | 0.034 |
| along_pool_plane | 0.910 ± 0.018 | -8.0 | 0.037 |
| cell_plane | 0.873 ± 0.023 | -11.3 | 0.046 |
| cell_along_plane | 0.905 ± 0.021 | -8.6 | 0.042 |
| cell_pool_plane | 1.002 ± 0.019 | +0.2 | 0.037 |
| cell_along_pool_plane | 0.985 ± 0.020 | -1.3 | 0.040 |
| cellq_plane | 0.875 ± 0.022 | -10.9 | 0.044 |
| cellq_along_plane | 0.907 ± 0.021 | -8.1 | 0.043 |
| cellq_pool_plane | 1.007 ± 0.019 | +0.6 | 0.038 |
| cellq_along_pool_plane | 0.989 ± 0.020 | -0.9 | 0.040 |
| cellxv_plane | 0.865 ± 0.038 | -6.6 | 0.076 |
| cellxv_along_plane | 0.890 ± 0.025 | -7.5 | 0.050 |
| cellxv_pool_plane | 0.970 ± 0.018 | -2.8 | 0.036 |
| cellxv_along_pool_plane | 0.951 ± 0.019 | -4.3 | 0.039 |

## af_drag_radius (airfoil Kn0.39 diffuse, radius)

| variant | base ratio | t | spread | pool1 ratio | t | spread | pool2 ratio | t | spread |
|---|---|---|---|---|---|---|---|---|---|
| one | -0.837 ± 2.513 | -1.5 | 7.109 | 0.824 ± 0.453 | -0.7 | 1.282 | 0.815 ± 0.603 | -0.6 | 1.705 |
| one_pool | 0.956 ± 0.582 | -0.2 | 1.648 | 0.995 ± 0.136 | -0.1 | 0.384 | 1.003 ± 0.088 | +0.0 | 0.249 |
| along | 0.222 ± 0.833 | -1.9 | 2.355 | 1.109 ± 0.222 | +0.9 | 0.627 | 1.112 ± 0.244 | +0.9 | 0.690 |
| along_pool | 0.604 ± 0.620 | -1.3 | 1.753 | 1.192 ± 0.117 | +2.9 | 0.331 | 1.242 ± 0.090 | +3.6 | 0.255 |
| one_pool_plane | 0.960 ± 0.581 | -0.1 | 1.645 | 1.004 ± 0.137 | +0.1 | 0.387 | 1.004 ± 0.088 | +0.1 | 0.250 |
| along_pool_plane | 0.606 ± 0.619 | -1.3 | 1.752 | 1.196 ± 0.118 | +3.0 | 0.333 | 1.243 ± 0.090 | +3.7 | 0.255 |
| cell_plane | 1.453 ± 0.612 | +1.5 | 1.730 | 1.202 ± 0.136 | +2.7 | 0.384 | 1.171 ± 0.149 | +2.2 | 0.421 |
| cell_along_plane | 1.019 ± 0.272 | +0.1 | 0.768 | 1.339 ± 0.079 | +6.3 | 0.223 | 1.328 ± 0.087 | +5.8 | 0.245 |
| cell_pool_plane | 1.073 ± 0.221 | +0.6 | 0.626 | 1.031 ± 0.079 | +0.6 | 0.223 | 0.943 ± 0.082 | -1.0 | 0.231 |
| cell_along_pool_plane | 0.859 ± 0.224 | -1.3 | 0.633 | 1.244 ± 0.066 | +4.7 | 0.188 | 1.220 ± 0.074 | +3.9 | 0.208 |
| cellq_plane | 1.464 ± 0.611 | +1.5 | 1.728 | 1.201 ± 0.136 | +2.7 | 0.385 | 1.170 ± 0.149 | +2.2 | 0.423 |
| cellq_along_plane | 1.021 ± 0.271 | +0.2 | 0.767 | 1.339 ± 0.079 | +6.3 | 0.223 | 1.328 ± 0.087 | +5.8 | 0.246 |
| cellq_pool_plane | 1.187 ± 0.231 | +1.5 | 0.655 | 1.028 ± 0.075 | +0.5 | 0.213 | 0.947 ± 0.073 | -0.9 | 0.205 |
| cellq_along_pool_plane | 0.869 ± 0.216 | -1.2 | 0.610 | 1.235 ± 0.066 | +4.4 | 0.187 | 1.214 ± 0.071 | +3.8 | 0.201 |
| cellxv_plane | 1.397 ± 0.550 | +1.4 | 1.555 | 1.179 ± 0.123 | +2.8 | 0.348 | 1.127 ± 0.140 | +1.8 | 0.396 |
| cellxv_along_plane | 1.022 ± 0.272 | +0.2 | 0.770 | 1.346 ± 0.074 | +6.6 | 0.210 | 1.329 ± 0.080 | +6.0 | 0.227 |
| cellxv_pool_plane | 1.106 ± 0.253 | +0.7 | 0.715 | 1.085 ± 0.092 | +1.4 | 0.260 | 1.019 ± 0.085 | +0.3 | 0.239 |
| cellxv_along_pool_plane | 0.771 ± 0.223 | -2.1 | 0.631 | 1.197 ± 0.070 | +3.6 | 0.198 | 1.170 ± 0.071 | +2.9 | 0.200 |

## af_drag_mode2 (airfoil Kn0.39 diffuse, cos2θ)

| variant | base ratio | t | spread | pool1 ratio | t | spread |
|---|---|---|---|---|---|---|
| one | 2.268 ± 1.859 | -1.4 | 5.258 | 1.323 ± 0.307 | -2.1 | 0.869 |
| one_pool | 0.782 ± 0.590 | +0.7 | 1.669 | 1.145 ± 0.103 | -2.1 | 0.292 |
| along | 1.509 ± 0.507 | -2.0 | 1.435 | 1.199 ± 0.133 | -2.9 | 0.377 |
| along_pool | 1.008 ± 0.496 | -0.0 | 1.404 | 1.118 ± 0.082 | -1.9 | 0.233 |
| one_pool_plane | 0.786 ± 0.591 | +0.7 | 1.672 | 1.151 ± 0.105 | -2.2 | 0.298 |
| along_pool_plane | 1.011 ± 0.496 | -0.0 | 1.404 | 1.122 ± 0.082 | -2.0 | 0.233 |
| cell_plane | 0.858 ± 0.547 | +0.5 | 1.547 | 1.008 ± 0.084 | -0.1 | 0.236 |
| cell_along_plane | 1.170 ± 0.171 | -1.6 | 0.484 | 1.109 ± 0.046 | -2.0 | 0.131 |
| cell_pool_plane | 1.153 ± 0.224 | -1.2 | 0.633 | 1.184 ± 0.053 | -3.3 | 0.151 |
| cell_along_pool_plane | 1.281 ± 0.151 | -2.9 | 0.428 | 1.166 ± 0.035 | -3.1 | 0.100 |
| cellq_plane | 0.846 ± 0.546 | +0.5 | 1.546 | 1.009 ± 0.084 | -0.2 | 0.237 |
| cellq_along_plane | 1.168 ± 0.171 | -1.6 | 0.483 | 1.109 ± 0.046 | -2.0 | 0.131 |
| cellq_pool_plane | 1.072 ± 0.229 | -0.6 | 0.647 | 1.207 ± 0.051 | -3.7 | 0.144 |
| cellq_along_pool_plane | 1.276 ± 0.152 | -2.8 | 0.431 | 1.175 ± 0.037 | -3.2 | 0.104 |
| cellxv_plane | 0.879 ± 0.481 | +0.5 | 1.361 | 1.014 ± 0.064 | -0.2 | 0.182 |
| cellxv_along_plane | 1.170 ± 0.164 | -1.6 | 0.463 | 1.096 ± 0.042 | -1.7 | 0.119 |
| cellxv_pool_plane | 0.976 ± 0.271 | +0.2 | 0.767 | 1.183 ± 0.064 | -3.3 | 0.180 |
| cellxv_along_pool_plane | 1.244 ± 0.158 | -2.4 | 0.447 | 1.153 ± 0.039 | -2.8 | 0.110 |

## Scorecard (base field over every gate; pool1 over the gates that ran it)

| variant | basis | gates | RMS(ratio - 1) | max abs(ratio - 1) | RMS t | median spread | airfoil spread (r, m2) |
|---|---|---|---|---|---|---|---|
| one | base | 24 | nan | nan | nan | nan | 7.11 / 5.26 |
| one_pool | base | 24 | nan | nan | nan | nan | 1.65 / 1.67 |
| along | base | 24 | nan | nan | nan | nan | 2.36 / 1.43 |
| along_pool | base | 24 | nan | nan | nan | nan | 1.75 / 1.40 |
| one_pool_plane | base | 24 | 0.136 | 0.431 | 5.0 | 0.038 | 1.64 / 1.67 |
| along_pool_plane | base | 24 | 0.122 | 0.394 | 4.1 | 0.038 | 1.75 / 1.40 |
| cell_plane | base | 24 | nan | nan | nan | nan | 1.73 / 1.55 |
| cell_along_plane | base | 24 | nan | nan | nan | nan | 0.77 / 0.48 |
| cell_pool_plane | base | 24 | 0.144 | 0.402 | 6.3 | 0.046 | 0.63 / 0.63 |
| cell_along_pool_plane | base | 24 | 0.106 | 0.281 | 4.1 | 0.040 | 0.63 / 0.43 |
| cellq_plane | base | 24 | nan | nan | nan | nan | 1.73 / 1.55 |
| cellq_along_plane | base | 24 | nan | nan | nan | nan | 0.77 / 0.48 |
| cellq_pool_plane | base | 24 | nan | nan | nan | nan | 0.65 / 0.65 |
| cellq_along_pool_plane | base | 24 | nan | nan | nan | nan | 0.61 / 0.43 |
| cellxv_plane | base | 24 | nan | nan | nan | nan | 1.56 / 1.36 |
| cellxv_along_plane | base | 24 | nan | nan | nan | nan | 0.77 / 0.46 |
| cellxv_pool_plane | base | 24 | nan | nan | nan | nan | 0.72 / 0.77 |
| cellxv_along_pool_plane | base | 24 | 0.112 | 0.244 | 3.5 | 0.039 | 0.63 / 0.45 |
| one | pool1 | 5 | 0.175 | 0.323 | 3.6 | 0.367 | 1.28 / 0.87 |
| one_pool | pool1 | 5 | 0.102 | 0.145 | 5.7 | 0.066 | 0.38 / 0.29 |
| along | pool1 | 5 | 0.119 | 0.199 | 4.5 | 0.168 | 0.63 / 0.38 |
| along_pool | pool1 | 5 | 0.115 | 0.192 | 4.7 | 0.062 | 0.33 / 0.23 |
| one_pool_plane | pool1 | 5 | 0.100 | 0.151 | 5.3 | 0.066 | 0.39 / 0.30 |
| along_pool_plane | pool1 | 5 | 0.116 | 0.196 | 4.3 | 0.063 | 0.33 / 0.23 |
| cell_plane | pool1 | 5 | 0.127 | 0.202 | 4.4 | 0.149 | 0.38 / 0.24 |
| cell_along_plane | pool1 | 5 | 0.169 | 0.339 | 5.2 | 0.077 | 0.22 / 0.13 |
| cell_pool_plane | pool1 | 5 | 0.136 | 0.184 | 7.6 | 0.065 | 0.22 / 0.15 |
| cell_along_pool_plane | pool1 | 5 | 0.146 | 0.244 | 5.4 | 0.064 | 0.19 / 0.10 |
| cellq_plane | pool1 | 5 | 0.127 | 0.201 | 4.2 | 0.144 | 0.38 / 0.24 |
| cellq_along_plane | pool1 | 5 | 0.170 | 0.339 | 5.4 | 0.076 | 0.22 / 0.13 |
| cellq_pool_plane | pool1 | 5 | 0.140 | 0.207 | 7.5 | 0.065 | 0.21 / 0.14 |
| cellq_along_pool_plane | pool1 | 5 | 0.147 | 0.235 | 5.5 | 0.064 | 0.19 / 0.10 |
| cellxv_plane | pool1 | 5 | 0.110 | 0.179 | 3.8 | 0.132 | 0.35 / 0.18 |
| cellxv_along_plane | pool1 | 5 | 0.169 | 0.346 | 5.2 | 0.071 | 0.21 / 0.12 |
| cellxv_pool_plane | pool1 | 5 | 0.113 | 0.183 | 5.0 | 0.074 | 0.26 / 0.18 |
| cellxv_along_pool_plane | pool1 | 5 | 0.122 | 0.197 | 4.6 | 0.063 | 0.20 / 0.11 |

## Terms of cell_pool_plane (base) / mean FD

| gate | pathwise_x | pathwise_v | pathwise_tally | score | timing | silhouette | order | volume |
|---|---|---|---|---|---|---|---|---|
| sq_spec | +0.012 | -0.051 | +1.030 | +0.000 | +0.001 | +0.000 | +0.007 | -0.012 |
| sq_diff | -0.036 | +0.000 | +0.496 | -0.060 | +0.005 | +0.489 | +0.012 | -0.035 |
| c10_spec | +0.215 | -0.104 | +1.240 | +0.000 | -0.001 | +0.000 | +0.004 | -0.151 |
| c10_spec_dt2 | +0.165 | -0.061 | +1.222 | +0.000 | -0.010 | +0.000 | +0.004 | -0.140 |
| c10_spec_g2n | +0.102 | -0.104 | +1.202 | +0.000 | +0.000 | +0.000 | +0.005 | -0.109 |
| c10_spec_g4n | -0.045 | -0.076 | +1.234 | +0.000 | +0.001 | +0.000 | -0.012 | -0.007 |
| c10_spec_2dt | +0.233 | -0.141 | +1.214 | +0.000 | -0.001 | +0.000 | -0.002 | -0.155 |
| c10_spec_h1000 | +0.259 | -0.153 | +1.224 | +0.000 | -0.001 | +0.000 | +0.007 | -0.155 |
| c10_spec_mem | +0.205 | -0.102 | +1.224 | +0.000 | -0.001 | +0.000 | +0.006 | -0.147 |
| c10_spec_gz001 | +0.205 | -0.102 | +1.225 | +0.000 | -0.001 | +0.000 | +0.006 | -0.147 |
| c10_spec_two | +0.203 | -0.103 | +1.224 | +0.000 | -0.001 | +0.000 | -0.006 | -0.147 |
| c10_spec_v816 | +0.206 | -0.102 | +1.224 | +0.000 | -0.001 | +0.000 | +0.001 | -0.147 |
| c10_spec_kn2 | +0.058 | -0.024 | +1.050 | +0.000 | +0.003 | +0.000 | +0.001 | -0.034 |
| c10_spec_kn10 | +0.008 | -0.003 | +1.009 | +0.000 | +0.002 | +0.000 | +0.000 | -0.007 |
| c10_diff | -0.247 | +0.000 | +0.099 | -0.024 | -0.021 | +0.914 | +0.354 | -0.086 |
| c10_diff_dt2 | -0.439 | +0.000 | +0.099 | -0.017 | -0.017 | +0.860 | +0.313 | +0.167 |
| c10_diff_f2 | -0.241 | +0.000 | +0.098 | -0.023 | -0.003 | +0.862 | +0.320 | -0.085 |
| c10_diff_g2n | -0.241 | +0.000 | +0.101 | -0.024 | -0.002 | +0.873 | +0.371 | +0.323 |
| c10_diff_mem | -0.247 | +0.000 | +0.099 | -0.024 | -0.021 | +0.914 | +0.354 | -0.086 |
| c10_diff_h600 | -0.223 | +0.000 | +0.099 | -0.039 | -0.021 | +0.914 | +0.373 | -0.171 |
| c10_diff_kn2 | -0.083 | +0.000 | +0.084 | -0.012 | +0.004 | +0.930 | +0.088 | -0.008 |
| c10_diff_kn10 | -0.017 | +0.000 | +0.074 | -0.003 | +0.003 | +0.928 | +0.006 | +0.013 |
| af_drag_radius | -0.523 | +0.000 | +0.290 | -0.036 | -0.002 | +0.853 | +0.144 | +0.347 |
| af_drag_mode2 | +0.404 | -0.000 | +0.556 | -0.136 | -0.019 | +0.595 | -0.047 | -0.201 |
