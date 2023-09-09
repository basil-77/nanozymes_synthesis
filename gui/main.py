import numpy
import numpy as np
import sys


import ns

from PyQt5 import QtWidgets
from nonameteam import Ui_MainWindow

with open("formulas.txt") as f:

    mylist = list()
    for i in f.readlines():
        mylist.append(i.strip())
    f.close()




ns_main = ns.nanozymes_synthesis()
ns_main.load('peroxidase.xlsx',
             'nanozymes_add.xlsx',
             'nanozymes_formulas.xlsx',
             'reaction_types.xlsx',
             'Rs.xlsx',
             'solvents.xlsx',
             'solvents_add.xlsx'
                         )
ns_main.model_load('saved', 'x_scaler')


class mywindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(mywindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.comboBox.addItems(sorted(mylist))
        self.ui.pushButton.clicked.connect(self.prognoseButtonClicked)



    #передать параметры в модель не забыть
    def prognoseButtonClicked(self):
        formula = self.ui.comboBox.currentText() #text
        length = self.ui.lineEdit.text()         #text
        width = self.ui.lineEdit_2.text()       #text
        depth = self.ui.lineEdit_3.text()       #text
        print(formula)                          #text
        encode = ns_main.encode_nanozymes(formula) #int
        list2 = encode.tolist()
        list1 = [length, width, depth]
        tonums = list(map(float, list1))
        joinedlist = tonums+list2
        np_array = numpy.array([joinedlist])
        print(np_array.dtype)
        print(np_array)

        res = ns_main.predict(np_array)


        print(numpy.shape(np_array))
        print(encode)
        print('strt')
        print('Done')
        print(res)
        self.ui.textEdit.setPlainText(str(res))




app = QtWidgets.QApplication([])
application = mywindow()
application.show()
sys.exit(app.exec())
