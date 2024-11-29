rm -rf dist
rm -rf python/target
mkdir -p python/target
cd python/target
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8
cd ../..
python python/setup.py bdist_wheel
cd python/target
python -m venv venv_test

pip install ../../dist/*.whl
cd ..
python examples/simple.py