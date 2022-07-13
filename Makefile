CC       = nvcc 
LIBS     = -lcublas -lcusparse
EXE      = cuda_trainer
SRC      = $(wildcard *.cu */*.cu */*/*.cu)
FLAGS    = -dlto -Xcompiler "-march=native -Ofast"
default:
	$(CC) $(FLAGS) $(SRC) $(LIBS) -o $(EXE)