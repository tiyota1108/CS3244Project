TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
#NSYNC_INC=$TF_INC"/external/nsync/public"
CUDA_PATH=/usr/local/cuda/bin
CXXFLAGS=''


if [[ "$OSTYPE" =~ ^darwin ]]; then
	CXXFLAGS+='-undefined dynamic_lookup'
fi

cd roi_pooling_layer

if [ -d "$CUDA_PATH" ]; then
	nvcc -std=c++11 -c -o roi_pooling_op.cu.o roi_pooling_op_gpu.cu.cc \
		-I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CXXFLAGS \
		-arch=sm_37 -L$TF_LIB -ltensorflow_framework \
                -D_GLIBCXX_USE_CXX11ABI=0

	g++ -std=c++11 -shared -o roi_pooling.so roi_pooling_op.cc \
		roi_pooling_op.cu.o -I $TF_INC  -D GOOGLE_CUDA=1 -fPIC $CXXFLAGS \
		-lcudart -L $CUDA_PATH/lib64 \
		-D_GLIBCXX_USE_CXX11_ABI=0
else
	g++ -std=c++11 -shared -o roi_pooling.so roi_pooling_op.cc \
		-I $TF_INC  -fPIC $CXXFLAGS
fi

cd ..

#cd feature_extrapolating_layer

#nvcc -std=c++11 -c -o feature_extrapolating_op.cu.o feature_extrapolating_op_gpu.cu.cc \
#	-I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_50

#g++ -std=c++11 -shared -o feature_extrapolating.so feature_extrapolating_op.cc \
#	feature_extrapolating_op.cu.o -I $TF_INC -fPIC -lcudart -L $CUDA_PATH/lib64
#cd ..

