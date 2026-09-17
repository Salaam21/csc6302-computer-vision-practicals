"""Run all Practical 6 tasks in order."""
from __future__ import annotations

from prepare_assets import main as prepare
from task1_optical_flow import run as t1
from task2_kalman import run as t2
from task3_calibrate import run as t3
from task4_ar import run as t4
from task5_stereo import run as t5


def main():
    prepare()
    t1()
    t2()
    t3()
    t4()
    t5()
    print("Practical 6 complete.")


if __name__ == "__main__":
    main()
