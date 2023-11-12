#ifndef CELERITAS_STORAGE_H
#define CELERITAS_STORAGE_H

#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include "batch.h"
#include "buffer.h"
#include "datatypes.h"

using std::vector;
using std::string;
using std::shared_ptr;
using std::list;
using std::unordered_map;

#define MAX_SHUFFLE_SIZE 4E8
#define MAX_SORT_SIZE 4E8

void renameFile(string old_filename, string new_filename);

void copyFile(string src_filename, string dst_filename);

class Storage {
  public:
    int64_t dim0_size_;
    int64_t dim1_size_;
    torch::Dtype dtype_;
    bool initialized_;
    vector<int64_t> edge_bucket_sizes_;
    torch::Tensor data_;

    virtual ~Storage() {};

    virtual torch::Tensor indexRead(Indices indices) = 0;

    virtual void indexAdd(Indices indices, torch::Tensor values) = 0;

    virtual torch::Tensor range(int64_t offset, int64_t n) = 0;

    virtual void indexPut(Indices indices, torch::Tensor values) = 0;

    virtual void load() = 0;

    virtual void unload(bool write = false) = 0;

    virtual void checkpoint(int epoch_id) = 0;

    virtual void shuffle() = 0;

    virtual void sort(bool src) = 0;

    int64_t getDim0() {
        return dim0_size_;
    }

    bool isInitialized() {
        return initialized_;
    }

    void setInitialized(bool init) {
        initialized_ = init;
    }

    void readPartitionSizes(string filename) {
        std::ifstream partition_file(filename);
        edge_bucket_sizes_.clear();
        int64_t size;
        while (partition_file >> size) {
            edge_bucket_sizes_.push_back(size);
        }
    }

    vector<int64_t> getEdgeBucketSizes() {
        return edge_bucket_sizes_;
    }
};

class PartitionBufferStorage : public Storage {
  protected:
    string filename_;

    bool loaded_;

    PartitionBuffer *buffer_;

    shared_ptr<PartitionBufferOptions> options_;

  public:
    PartitionBufferStorage(string filename, int64_t dim0_size, int64_t dim1_size, shared_ptr<PartitionBufferOptions> options);

    PartitionBufferStorage(string filename, torch::Tensor data, shared_ptr<PartitionBufferOptions> options);

    PartitionBufferStorage(string filename, shared_ptr<PartitionBufferOptions> options);

    ~PartitionBufferStorage();

    void rangePut(int64_t offset, torch::Tensor values);

    void append(torch::Tensor values);

    void load() override;

    void unload(bool write) override;

    void checkpoint(int epoch_id) override;

    torch::Tensor indexRead(Indices indices) override;

    void indexAdd(Indices indices, torch::Tensor values) override;

    torch::Tensor range(int64_t offset, int64_t n) override;

    void indexPut(Indices indices, torch::Tensor values) override;

    void shuffle() override;

    void sort(bool src) override;

    Indices getRandomIds(int64_t size) {
        return buffer_->getRandomIds(size);
    }

    bool hasSwap() {
        return buffer_->hasSwap();
    }

    void performNextSwap() {
        buffer_->performNextSwap();
    }

    torch::Tensor getGlobalToLocalMap(bool get_current) {
        return buffer_->getGlobalToLocalMap(get_current);
    }

    void sync() {
        buffer_->sync();
    }

    void setBufferOrdering(vector<torch::Tensor> buffer_states) {
        buffer_->setBufferOrdering(buffer_states);
    }

    std::vector<int> getNextAdmit() {
        return buffer_->getNextAdmit();
    }

    std::vector<int> getNextEvict() {
        return buffer_->getNextEvict();
    }

    int64_t getNumInMemory() {
        return buffer_->getNumInMemory();
    }

};

class FlatFile : public Storage {
  private:
    string filename_;

    int fd_;

    bool loaded_;


  public:
    FlatFile(string filename, int64_t dim0_size, int64_t dim1_size, torch::Dtype dtype);

    FlatFile(string filename, torch::Tensor data);

    FlatFile(string filename, torch::Dtype dtype);

    ~FlatFile() {};

    void rangePut(int64_t offset, torch::Tensor values);

    void append(torch::Tensor values);

    void load() override;

    void unload(bool write) override;

    void checkpoint(int epoch_id) override;

    torch::Tensor indexRead(Indices indices) override;

    void indexAdd(Indices indices, torch::Tensor values) override;

    torch::Tensor range(int64_t offset, int64_t n) override;

    void indexPut(Indices indices, torch::Tensor values) override;

    void shuffle() override;

    void sort(bool src) override;

    void move(string new_filename);

    void copy(string new_filename, bool rename);

    void mem_load();

    void mem_unload(bool write);
};

class MemoryMap : public Storage {
private:
    string filename_;

    int fd_;

    bool loaded_;

    int block_size_;

public:
    MemoryMap(string filename, int64_t dim0_size, int64_t dim1_size, torch::Dtype dtype);

    MemoryMap(string filename, torch::Tensor data);

    MemoryMap(string filename, torch::Dtype dtype);

    void rangePut(int64_t offset, torch::Tensor values);

    void load() override;

    void unload(bool write) override;

    void checkpoint(int epoch_id) override;

    torch::Tensor indexRead(Indices indices) override;

    void indexAdd(Indices indices, torch::Tensor values) override;

    torch::Tensor range(int64_t offset, int64_t n) override;

    void mem_load();

    void mem_unload(bool write);

    void shuffle() override;
};

/** In memory storage for data which fits in either GPU or CPU memory. */
class InMemory : public Storage {
  private:
    string filename_;

    int fd_;

    bool loaded_;

    torch::DeviceType device_;

    torch::Tensor src_sorted_list_;

    torch::Tensor dst_sorted_list_;

  public:

    InMemory(string filename, int64_t dim0_size, int64_t dim1_size, torch::Dtype dtype, torch::DeviceType device);

    InMemory(string filename, torch::Tensor data, torch::DeviceType device);

    InMemory(string filename, torch::Dtype dtype);

    ~InMemory() {};

    void load() override;

    void unload(bool write) override;

    void checkpoint(int epoch_id) override;

    torch::Tensor indexRead(Indices indices) override;

    void indexAdd(Indices indices, torch::Tensor values) override;

    torch::Tensor range(int64_t offset, int64_t n) override;

    void indexPut(Indices indices, torch::Tensor values) override;

    void force_load();

    void shuffle() override;

    void sort(bool src) override;
};

#endif 
