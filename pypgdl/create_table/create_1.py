from pypgdl import *

name = 'mnist'

pl = start()

pl.create_table_x(name + "_x_train")
pl.create_table_x(name + "_x_test")
pl.create_table_y(name + "_y_train")
pl.create_table_y(name + "_y_test")

end(pl)
