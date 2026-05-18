from base import base
from speed_check import speed_check
from weights_change import weights_change
from batch_size import batch_size

choice = int(input("выберите цифру. 1 - базовая работа, 2 - изучение learning rate, 3 - изучение batc size, 4 - изучение weights"))
if choice == 1:
    base()
elif choice == 2:
    speed_check()
elif choice == 3:
    batch_size()
elif choice == 4:
     weights_change()
else:
    print("Неверный ввод")