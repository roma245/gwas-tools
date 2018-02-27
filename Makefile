GENERATE_SUBSETS_SOURCE=generate_subsets.cpp
GENERATE_SUBSETS_BINARY=generate_subsets.so
PYGCO_PATH=pygco
PYGCO_BINARY=$(PYGCO_PATH)/cgco.o
CACHE_CATALOGS=cache/xgboost cache/frn
EXPERIMENTS_RESULTS_CATALOG=experiments_results

all: $(GENERATE_SUBSETS_BINARY) $(PYGCO_BINARY) $(EXPERIMENTS_RESULTS_CATALOG) $(CACHE_CATALOGS)


$(GENERATE_SUBSETS_BINARY): $(GENERATE_SUBSETS_SOURCE)
	g++ $(GENERATE_SUBSETS_SOURCE) --std=c++11 -o $(GENERATE_SUBSETS_BINARY) -shared -fPIC -O2 -lboost_system -lboost_python -lboost_numpy -I /usr/include/python2.7

$(PYGCO_BINARY): $(PYGCO_PATH)
	make -C $(PYGCO_PATH)
$(CACHE_CATALOGS): 
	rm -r -f cache
	mkdir cache
	mkdir cache/xgboost
	mkdir cache/frn
$(EXPERIMENTS_RESULTS_CATALOG):
	mkdir $(EXPERIMENTS_RESULTS_CATALOG)
clean:
	rm -f $(GENERATE_SUBSETS_BINARY)
	rm -f $(PYGCO_PATH)/*.so
	rm -f $(PYGCO_PATH)/*.o
