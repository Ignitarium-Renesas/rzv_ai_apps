rm -rf build
mkdir -p build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=./toolchain/runtime.cmake ..
make -j$(nproc)





