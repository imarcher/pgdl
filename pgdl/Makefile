

EXTENSIONDIR = /usr/local/pgsql/share/extension
SODIR = /usr/local/pgsql/lib

PG_FLAGS = -I /usr/local/pgsql/include/server
DNN_LIB = -L /usr/local/lib -ldnnl
THREAD_LIB = -L -lpthread -lm
DNN_HEADER = -I /usr/local/include
DNN_METHOD = -I /home/hh/oneDNN/examples

OBJS_CC = $(subst .cpp,.o,$(wildcard *.cpp))

TARGET = pgdl.so pgdl.o $(OBJS_CC)

all: $(TARGET)


pgdl.so: pgdl.o $(OBJS_CC)
	g++ -shared -o $@ $^ $(DNN_LIB) $(DNN_METHOD) $(PG_FLAGS) $(DNN_HEADER) $(THREAD_LIB)

pgdl.o: pgdl.c
	gcc -fPIC -c pgdl.c $(PG_FLAGS)

%.o: %.cpp
	g++ -fPIC -c $< $(PG_FLAGS) $(DNN_METHOD) $(DNN_HEADER) $(THREAD_LIB)


install:
	/bin/cp -f pgdl.so $(SODIR)/pgdl.so
	/bin/cp -f pgdl.control $(EXTENSIONDIR)/pgdl.control
	/bin/cp -f pgdl--1.0.sql $(EXTENSIONDIR)/pgdl--1.0.sql

uninstall:
	rm -f $(SODIR)/pgdl.so
	rm -f $(EXTENSIONDIR)/pgdl.control
	rm -f $(EXTENSIONDIR)/pgdl--1.0.sql

clean:
	rm -f $(TARGET)

del:
	rm -f $(SODIR)/pgdl.so
	rm -f $(EXTENSIONDIR)/pgdl.control
	rm -f $(EXTENSIONDIR)/pgdl--1.0.sql
	rm -f $(TARGET)










