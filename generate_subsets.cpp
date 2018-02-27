#include <vector>
#include <fstream>
#include <unordered_set>
#include <bitset>
#include <string>
#include <queue>
#include <sparsehash/sparse_hash_set>
#include <sparsehash/dense_hash_set>
#include <boost/numpy.hpp>
//#include <boost/python.hpp>

#include <iostream>

namespace bp = boost::python;
namespace bn = boost::numpy;

namespace NSubsetGenerator{

using std::vector;
const size_t MAX_BITSET_SIZE = 400; // количество образцо в выборке????
typedef std::bitset<MAX_BITSET_SIZE> bitset;
typedef vector<short> index_list;
typedef vector<vector<bool> > matrix;


template<typename T>
bn::ndarray vector_to_ndarray(const T &vector) {
    Py_intptr_t shape[1] = { vector.size() };
    bn::ndarray result = bn::zeros(
        1,
        shape,
        bn::dtype::get_builtin<typename T::value_type>()
    );
    std::copy(
        vector.begin(),
        vector.end(),
        reinterpret_cast<typename T::value_type*>(result.get_data())
    );
    return result;
}


matrix ndarray_to_matrix(const bn::ndarray& array) {
    bp::tuple shape = bp::extract<bp::tuple>(array.attr("shape"));
    size_t rows_count = bp::extract<size_t>(shape[0]);
    size_t columns_count = bp::extract<size_t>(shape[1]);
    matrix ans(rows_count, std::vector<bool> (columns_count));
    char* data_ptr = reinterpret_cast<char*>(array.get_data());

    for (size_t row_index = 0; row_index < rows_count; ++row_index) {
        for (size_t col_index = 0; col_index < columns_count; ++col_index) {
            ans[row_index][col_index] = data_ptr[rows_count * col_index + row_index];
        }
    }
    return ans;
}


index_list to_index_list(const bitset& bs) {
    index_list ans;
    for (size_t i = 0; i < bs.size(); ++i) {
        if (bs[i]) {
            ans.push_back(i);
        }
    }
    return ans;
}


bitset to_bitset(const index_list& ind_list) {
    bitset ans;
    for (int i = 0; i < ind_list.size(); ++i) {
        ans.set(ind_list[i]);
    }
    return ans;
}


template<typename T>
void clear_vector(vector<T>& to_clear) {
    vector<T> tmp;
    tmp.swap(to_clear);
}


class TPriorityBitsetComparator{
public:
    bool operator()(const std::pair<double, bitset>& a, const std::pair<double, bitset>& b) {
        if (a.first > b.first) {
            return true;
        } else if (b.first < a.first) {
            return false;
        } else {
            return a.second.count() > b.second.count();
        }

    }
};


class TSubsetGenerator{
    typedef google::sparse_hash_set<bitset, std::hash<bitset> > set;
    //typedef std::unordered_set<bitset> set;
    //typedef google::dense_hash_set<bitset, std::hash<bitset> > set;



    vector<bitset> sets;
    vector<bitset> sets_copy;

    matrix raw_matxix;

    vector<bitset> feature_bitsets;

    vector<bitset> get_simple_sets(const matrix& matr) const {
        vector<bitset> ans;

        size_t matr_size_x = matr.size();
        size_t matr_size_y = matr[0].size();

        for (size_t column = 0; column < matr_size_y; ++column) {
            index_list column_result;
            for (size_t row = 0; row < matr_size_x; ++row) {
                if (matr[row][column]) {
                    column_result.push_back(row);
                }
            }
            if (column_result.size() > 0) {
                ans.push_back(to_bitset(column_result));
            }
        }

        return ans;
    }

    void update(set& to_update, bitset element) const {
        bitset empty;

        set elements_to_add;
        //elements_to_add.set_deleted_key(bitset());
        //elements_to_add.set_empty_key(empty);

        if (element != empty) {
            elements_to_add.insert(element);
        }

        for (auto it = to_update.begin(); it != to_update.end(); ++it) {
            bitset intersection = *it;
            intersection &= element;
            if (intersection != empty) {
                elements_to_add.insert(intersection);
            }
        }

        for (auto it = elements_to_add.begin(); it != elements_to_add.end(); ++it) {
            auto element = *it;
            to_update.insert(element);
        }
    }

    void generate_and_set_matrix(const matrix& matr) {
        clear_vector(sets);
        std::cout << matr.size() << ' ' << matr[0].size() << std::endl;
        vector<bitset> simple_sets = get_simple_sets(matr);
        std::cout << simple_sets.size() << std::endl;
        if (simple_sets.size() != matr[0].size()) throw 1;

        set result_set;
        //result_set.set_deleted_key(bitset());
        //result_set.set_empty_key(bitset());


        for (size_t i = 0; i < simple_sets.size(); ++i) {
            std::cout << i << ' ' << result_set.size() << std::endl;
            update(result_set, simple_sets[i]);
        }

        sets = vector<bitset>(result_set.begin(), result_set.end());

        /*for(auto it = result_set.begin(); it != result_set.end(); ++it) {
            auto element = *it;
            sets.push_back(element);
            //result_set.erase(element);
        }*/
    }

    std::pair<bitset, bitset> get_y_and_indexes_mask(bn::ndarray y, bn::ndarray indexes) {
        bp::tuple shape = bp::extract<bp::tuple>(y.attr("shape"));
        int size = bp::extract<int>(shape[0]);
        bitset y_mask;
        bitset indexes_mask;
        for (size_t i = 0; i < size; ++i) {
            int y_val = bp::extract<int>(y[i]);
            int ind_val = bp::extract<int>(indexes[i]);
            indexes_mask.set(ind_val);
            if (y_val) {
                y_mask.set(ind_val);
            }
        }
        return std::make_pair(y_mask, indexes_mask);
    }

public:
    TSubsetGenerator() {
    }

    void generate_and_set(const bn::ndarray& input_matrix) {
        generate_and_set_matrix(ndarray_to_matrix(input_matrix));
    }

    void store(std::string filename) const {
        std::ofstream fout(filename);
        fout << sets.size() << '\n';
        for (auto it = sets.begin(); it != sets.end(); ++it) {
            index_list set_index_list = to_index_list(*it);
            fout << set_index_list.size() << ' ';
            for (size_t i = 0; i < set_index_list.size(); ++i) {
                fout << set_index_list[i] << ' ';
            }
            fout << '\n';
        }
        fout.flush();
        fout.close();
    }

    void load(std::string filename) {
        clear_vector(sets);
        std::ifstream fin(filename);
        size_t sets_size;
        fin >> sets_size;
        sets.reserve(sets_size);
        for (size_t set_index = 0; set_index < sets_size; ++set_index) {
            size_t set_size;
            fin >> set_size;
            index_list set_index_list;
            for (size_t i = 0; i < set_size; ++i) {
                size_t element_index;
                fin >> element_index;
                set_index_list.push_back(element_index);
            }
            sets.push_back(to_bitset(set_index_list));
        }
    }

    size_t get_sets_count() const {
        return sets.size();
    }

    void restore() {
        if (sets_copy.size() > 0) {
            sets.swap(sets_copy);
            sets_copy.clear();
        }
    }

    void set_filtered_min_size(int min_size) {
        std::cout << "set_filtered_min_size " << min_size << std::endl;
        sets_copy.clear();
        for (size_t i = 0; i < sets.size(); ++i) {
            if (to_index_list(sets[i]).size() >= min_size) {
                sets_copy.push_back(sets[i]);
            }
        }
        sets.swap(sets_copy);
        std::cout << "set_filtered_min_size done!!!" << min_size << std::endl;
    }

    void set_filtered_have_ones_in_positions(bn::ndarray positions) {
        bitset mask;
        bp::tuple shape = bp::extract<bp::tuple>(positions.attr("shape"));
        int size = bp::extract<int>(shape[0]);
        std::cout << "set_filtered_have_ones_in_positions " << size << std::endl;
        for (size_t i = 0; i < size; ++i) {
            int position = bp::extract<int>(positions[i]);
            mask.set(position);
        }
        sets_copy.clear();
        for (size_t i = 0; i < sets.size(); ++i) {
            if ((sets[i] & mask) == sets[i]) {
                sets_copy.push_back(sets[i]);
            }
        }
        sets.swap(sets_copy);
        std::cout << "set_filtered_have_ones_in_positions done!!!" << size << std::endl;
    }

    void set_filtered_best_beta_binomial(double alpha_regularization,
                                         double beta_regularization,
                                         bn::ndarray y,
                                         bn::ndarray indexes,
                                         int max_store) {

        std::cout << "set_filtered_best_beta_binomial " << max_store << std::endl;


        std::pair<bitset, bitset> y_and_indexes_mask = get_y_and_indexes_mask(y, indexes);
        bitset y_mask = y_and_indexes_mask.first;
        bitset indexes_mask = y_and_indexes_mask.second;

        sets_copy.clear();

        std::priority_queue<std::pair<double, bitset>,
                            vector<std::pair<double, bitset> >,
                            TPriorityBitsetComparator> result_queue;

        for (size_t i = 0; i < sets.size(); ++i) {
            int in_tran_all = (sets[i] & indexes_mask).count();
            int in_train_with_y_one = (sets[i] & y_mask).count();
            double alpha = alpha_regularization + in_train_with_y_one;
            double beta = beta_regularization + in_tran_all - in_train_with_y_one;
            double priority = (alpha - 1) / (alpha + beta - 2);
            result_queue.push(std::make_pair(priority, sets[i]));
            while (result_queue.size() > max_store) {
                result_queue.pop();
            }
        }

        sets_copy.resize(result_queue.size());
        size_t insert_index = result_queue.size();
        while (!result_queue.empty()) {
            --insert_index;
            sets_copy[insert_index] = result_queue.top().second;
            result_queue.pop();
        }
        sets.swap(sets_copy);
        std::cout << "set_filtered_best_beta_binomial done!!!" << max_store << std::endl;
    }

    bn::ndarray get_count_and_y_true_statistics(bn::ndarray y, bn::ndarray indexes) {
        std::pair<bitset, bitset> y_and_indexes_mask = get_y_and_indexes_mask(y, indexes);
        bitset y_mask = y_and_indexes_mask.first;
        bitset indexes_mask = y_and_indexes_mask.second;

        size_t max_stat_value = MAX_BITSET_SIZE;
        vector<vector<size_t> > statistics(max_stat_value + 1, vector<size_t>(max_stat_value + 1, 0));

        for (size_t i = 0; i < sets.size(); ++i) {
            int in_tran_all = (sets[i] & indexes_mask).count();
            int in_train_with_y_one = (sets[i] & y_mask).count();
            ++statistics[in_tran_all][in_train_with_y_one];
        }

        bn::ndarray statistics_ndarray = bn::zeros(bp::make_tuple(max_stat_value + 1, max_stat_value + 1),
                                                   bn::dtype::get_builtin<int>());

        for (size_t i = 0; i < statistics.size(); ++i) {
            for (size_t j = 0; j < statistics[i].size(); ++j) {
                statistics_ndarray[i][j] = statistics[i][j];
            }
        }

        return statistics_ndarray;
    }


    bn::ndarray get_set(size_t index) const {
        return vector_to_ndarray(to_index_list(sets[index]));
    }

    bitset get_feature_bitset(size_t feature_index) {
        bitset ans;
        for (size_t i = 0; i < raw_matxix.size(); ++i) {
            if (raw_matxix[i][feature_index]) {
                ans.set(i);
            }
        }
        return ans;
    }

    void set_raw_matrix(bn::ndarray matrix_ndarray) {
        raw_matxix = ndarray_to_matrix(matrix_ndarray);
        feature_bitsets.clear();
        for (size_t i = 0; i < raw_matxix[0].size(); ++i) {
            feature_bitsets.push_back(get_feature_bitset(i));
        }
    }

    vector<size_t> select_indexes(vector<size_t> possible_indexes,
                                  bitset result_mask,
                                  size_t checked_variants_limit
                                ) {
        vector<vector<size_t> > variants;
        variants.push_back(vector<size_t>());
        bitset ones_mask;

        for (size_t i = 0; i < MAX_BITSET_SIZE; ++i) {
            ones_mask.set(i);
        }

        for (size_t checked_variants = 0;
            checked_variants < variants.size();
            ++checked_variants) {
            bitset current_mask = ones_mask;
            vector<size_t> variant = variants[checked_variants];
            for (size_t i = 0; i < variant.size(); ++i) {
                current_mask &= feature_bitsets[possible_indexes[variant[i]]];
            }
            if (current_mask == result_mask) {
                vector<size_t> result;
                for (size_t i = 0; i < variant.size(); ++i) {
                    result.push_back(possible_indexes[variant[i]]);
                }
                return result;
            }
            size_t start_val = 0;
            if (variant.size() > 0) {
                start_val = variant[variant.size() - 1] + 1;
            }
            for (; start_val < possible_indexes.size(); ++start_val) {
                if (variants.size() < checked_variants_limit) {
                    vector<size_t> new_variant = variant;
                    new_variant.push_back(start_val);
                    variants.push_back(new_variant);
                } else {
                    break;
                }
            }
        }
        return vector<size_t>(1, MAX_BITSET_SIZE + 1000000);
    }


    bn::ndarray get_probable_features_indexes(bn::ndarray objects_indexes, int checked_variants_limit) {
        vector<size_t> possible_features_indexes;
        bp::tuple shape = bp::extract<bp::tuple>(objects_indexes.attr("shape"));
        size_t size = bp::extract<size_t>(shape[0]);
        vector<bool> possible_features_indexes_mask(raw_matxix[0].size(), true);
        bitset objects_indexes_mask;

        for (size_t i = 0; i < size; ++i) {
            size_t object_index = bp::extract<size_t>(objects_indexes[i]);
            objects_indexes_mask.set(object_index);
            for (size_t j = 0; j < raw_matxix[object_index].size(); ++j) {
                possible_features_indexes_mask[j] = possible_features_indexes_mask[j] & raw_matxix[object_index][j];
            }
        }
        for (size_t i = 0; i < possible_features_indexes_mask.size(); ++i) {
            if (possible_features_indexes_mask[i]) {
                possible_features_indexes.push_back(i);
            }
        }

        return vector_to_ndarray(select_indexes(possible_features_indexes, objects_indexes_mask, checked_variants_limit));
    }

};


} //namespace NSubsetGenerator


BOOST_PYTHON_MODULE(generate_subsets) {
    bn::initialize();
    bp::class_<NSubsetGenerator::TSubsetGenerator>("SubsetGenerator")
        .def("generate_and_set", &NSubsetGenerator::TSubsetGenerator::generate_and_set)
        .def("store", &NSubsetGenerator::TSubsetGenerator::store)
        .def("load", &NSubsetGenerator::TSubsetGenerator::load)
        .def("get_sets_count", &NSubsetGenerator::TSubsetGenerator::get_sets_count)
        .def("get_set", &NSubsetGenerator::TSubsetGenerator::get_set)
        .def("set_filtered_have_ones_in_positions", &NSubsetGenerator::TSubsetGenerator::set_filtered_have_ones_in_positions)
        .def("restore", &NSubsetGenerator::TSubsetGenerator::restore)
        .def("get_count_and_y_true_statistics", &NSubsetGenerator::TSubsetGenerator::get_count_and_y_true_statistics)
        .def("set_filtered_best_beta_binomial", &NSubsetGenerator::TSubsetGenerator::set_filtered_best_beta_binomial)
        .def("get_probable_features_indexes", &NSubsetGenerator::TSubsetGenerator::get_probable_features_indexes)
        .def("set_raw_matrix", &NSubsetGenerator::TSubsetGenerator::set_raw_matrix)
    ;
};
