curl -L $SECRET_CPLEX_LINK -o cplex.tar.gz
tar xvf cplex.tar.gz
if [[ $TRAVIS_PYTHON_VERSION == "3.4" ]]; then
	cd "cplex/python/3.4/x86-64_linux/";
fi
if [[ $TRAVIS_PYTHON_VERSION == "2.7" ]]; then
	cd "cplex/python/2.6/x86-64_linux/";
fi
pip install .
cd $TRAVIS_BUILD_DIR
