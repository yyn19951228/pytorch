#include <gtest/gtest.h>
#include <torch/torch.h>
#include <algorithm>
#include <memory>
#include <vector>

#include <test/cpp/api/support.h>

using namespace torch::nn;
using namespace torch::test;

struct ModuleDictTest : torch::test::SeedingFixture {};

TEST_F(ModuleDictTest, ContructsFromSharedPointer) {
  struct M : torch::nn::Module {
    explicit M(int value_) : value(value_) {}
    int value;
  };
  torch::OrderedDict<std::string, std::shared_ptr<Module>> dict{
      {"A", std::make_shared<M>(1)}, {"B", std::make_shared<M>(2)}};
  ModuleDict mDict(dict);
  ASSERT_EQ(mDict->size(), 2);
}

TEST_F(ModuleDictTest, ConstructsFromConcreteType) {
  static int copy_count;

  struct M : torch::nn::Module {
    explicit M(int value_) : value(value_) {}
    M(const M& other) : torch::nn::Module(other) {
      copy_count++;
    }
    int value;
  };

  torch::OrderedDict<std::string, Module> dict{
      {"A", M(1)}, {"B", M(2)}, {"C", M(3)}};
  ModuleDict mDict(dict);
  copy_count = 0;
  ASSERT_EQ(mDict->size(), 3);
  // Note: Avoid the copying when initalize the `ModuleDict`
  ASSERT_EQ(copy_count, 0);
}

TEST_F(ModuleDictTest, ConstructsFromModuleHolder) {
  struct MImpl : torch::nn::Module {
    explicit MImpl(int value_) : value(value_) {}
    int value;
  };

  struct M : torch::nn::ModuleHolder<MImpl> {
    using torch::nn::ModuleHolder<MImpl>::ModuleHolder;
    using torch::nn::ModuleHolder<MImpl>::get;
  };

  torch::OrderedDict<std::string, torch::nn::ModuleHolder<MImpl>> dict{
      {"A", M(1)}, {"B", M(2)}, {"C", M(3)}};
  ModuleDict mDict(dict);
  ASSERT_EQ(mDict->size(), 3);
}

TEST_F(ModuleDictTest, InsertAnElement) {
  struct M : torch::nn::Module {
    explicit M(int value_) : value(value_) {}
    int value;
  };

  ModuleDict mDict;
  ASSERT_EQ(mDict->size(), 0);
  ASSERT_TRUE(mDict->empty());
  mDict->insert("Linear", Linear(3, 4));
  ASSERT_EQ(mDict->size(), 1);
  mDict->insert("M1", std::make_shared<M>(1));
  ASSERT_EQ(mDict->size(), 2);
  mDict->insert("M2", M(2));
  ASSERT_EQ(mDict->size(), 3);
}

TEST_F(ModuleDictTest, Insertion) {
  struct MImpl : torch::nn::Module {
    explicit MImpl(int value_) : value(value_) {}
    int value;
  };
  TORCH_MODULE(M);

  ModuleDict mDict;
  mDict->insert("M1", MImpl(1));
  ASSERT_EQ(mDict->size(), 1);
  mDict->insert("M2", std::make_shared<MImpl>(2));
  ASSERT_EQ(mDict->size(), 2);
  mDict->insert("M3", M(3));
  ASSERT_EQ(mDict->size(), 3);
  mDict->insert("M4", M(4));
  ASSERT_EQ(mDict->size(), 4);
  ASSERT_EQ(mDict->get<MImpl>("M1").value, 1);
  ASSERT_EQ(mDict->get<MImpl>("M2").value, 2);
  ASSERT_EQ(mDict->get<MImpl>("M3").value, 3);
  ASSERT_EQ(mDict->get<MImpl>("M4").value, 4);

  std::unordered_map<std::string, size_t> U = {
      {"M1", 1}, {"M2", 2}, {"M3", 3}, {"M4", 4}};
  for (const auto& P : mDict->named_modules("", false))
    ASSERT_EQ(U[P.key()], P.value()->as<M>()->value);
}

TEST_F(ModuleDictTest, AccessWithAt) {
  struct M : torch::nn::Module {
    explicit M(int value_) : value(value_) {}
    int value;
  };
  torch::OrderedDict<std::string, std::shared_ptr<M>> dict{
      {"M1", std::make_shared<M>(1)},
      {"M2", std::make_shared<M>(2)},
      {"M3", std::make_shared<M>(3)}};
  std::vector<std::string> idx = {"M1", "M2", "M3"};
  ModuleDict mDict(dict);
  ASSERT_EQ(mDict->size(), 3);

  // returns the correct module for a given index
  for (size_t i = 0; i < idx.size(); ++i) {
    ASSERT_EQ(&mDict->get<M>(idx[i]), dict[idx[i]].get());
  }
}

TEST_F(ModuleDictTest, AccessWithPtr) {
  struct M : torch::nn::Module {
    explicit M(int value_) : value(value_) {}
    int value;
  };
  torch::OrderedDict<std::string, std::shared_ptr<M>> dict{
      {"M1", std::make_shared<M>(1)},
      {"M2", std::make_shared<M>(2)},
      {"M3", std::make_shared<M>(3)}};
  std::vector<std::string> idx = {"M1", "M2", "M3"};
  ModuleDict mDict(dict);
  ASSERT_EQ(mDict->size(), 3);

  // returns the correct module for a given index
  for (size_t i = 0; i < idx.size(); ++i) {
    ASSERT_EQ(mDict->ptr(idx[i]).get(), dict[idx[i]].get());
    ASSERT_EQ(mDict[idx[i]].get(), dict[idx[i]].get());
    ASSERT_EQ(mDict->ptr<M>(idx[i]).get(), dict[idx[i]].get());
  }
}

TEST_F(ModuleDictTest, SanityCheckForHoldingStandardModules) {
  // torch::OrderedDict<std::string, Module> dict{
  //     {"Linear", Linear(10, 3)},
  //     {"Conv1", Conv2d(1, 2, 3)},
  //     {"Dropout", Dropout(0.5)},
  //     {"BN", BatchNorm2d(5)},
  //     {"Embedding", Embedding(4, 10)},
  //     {"LSTM", LSTM(4, 5)}};
  torch::OrderedDict<std::string, AnyModule> dict{
      {"L", Linear(10, 3)}
      };
  // torch::OrderedDict<std::string, ModuleHolder<AnyModule>> dict{
  //     {"L", Linear(10, 3)}
  //     };
  ModuleDict mDict(dict);
  ASSERT_EQ(mDict->size(), 6);
}