import numpy as np 
from matplotlib import pyplot as plt 


if __name__ == "__main__":
    name = [
"myGEMM_no_kernel(double*, double*, double*, double*, double, double, int, int, int),",
"myGEMM_no_tB_kernel(double*, double*, double*, double*, double, double, int, int, int),",
"myGEMM_no_kernel(double*, double*, double*, double*, double, double, int, int, int),",
"myGEMM_no_na_tA_kernel(double*, double*, double*, double, int, int, int),",
"myGEMM_no_tB_kernel(double*, double*, double*, double*, double, double, int, int, int),",
"Dz1_schur_kernel(double*, double*, double*, int, int),",
"sigmoid_kernel(double*, double*, int, int),",
"softmax_kernel(double*, double*, double*, int, int, int),",
    ]
    duration = np.asarray([
78608835,
9341941,
1740697,
221082,
127780,
121261,
95831,
14304,
    ]) * 1e-3


    print("percent ", sum(duration[:3])/sum(duration))

    name = [n.split("(")[0] for n in name]
    pie_lab = [n if i < 3 else " " for i, n in enumerate(name) ]

    plt.figure()
    plt.pie(duration, labels=pie_lab, startangle=0)
    plt.axis('equal')
    plt.savefig("pie_chart.png", dpi=300)
