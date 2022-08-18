CC       = nvcc 
LIBS     = -lcublas 
EXE      = cuda_trainer
SRC      = $(wildcard *.cu */*.cu */*/*.cu)
FLAGS    = -Xcompiler "-march=native -Ofast"
default:
	$(CC) $(FLAGS) $(SRC) $(LIBS) -o $(EXE)