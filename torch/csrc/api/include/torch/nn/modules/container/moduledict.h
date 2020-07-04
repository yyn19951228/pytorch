#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/pimpl.h>
#include <torch/ordered_dict.h>
#include <vector>

namespace torch {
namespace nn {

/// A Dict of `Module`s that register its elements
///
/// \rst
/// .. code-block:: cpp
///
///   torch::nn::OrderedDict mInitDict {
///     {"Linear", torch::nn::Linear(3, 4),
///     {"BN", torch::nn::BatchNorm1d(4)},
///     {"Dropout", torch::nn::Dropout(0.5)}
///   };
///   torch::nn::ModuleDict mDict(mInitDict);
///
/// \endrst
///
/// Why should you use `ModuleDict` instead of a simple `OrderedDict`?
/// The value a `ModuleDict` provides over manually calling a sequence of
/// modules is that it allows treating the whole container *as a single module*,
/// such that performing a transformation on the `ModuleDict` applies to each of
/// the modules it stores (which are each a registered submodule of the
/// `ModuleList`). For example, calling
/// `.to(torch::kCUDA)` on a `ModuleDict` will move each module in the list to
/// CUDA memory. For example:
///
/// \rst
/// .. code-block:: cpp
///
///   torch::nn::OrderedDict mInitDict {
///     {"Linear", torch::nn::Linear(3, 4),
///     {"BN", torch::nn::BatchNorm1d(4)},
///     {"Dropout", torch::nn::Dropout(0.5)}
///   };
///   torch::nn::ModuleDict mDict(mInitDict);
///   mDict->to(torch::kCUDA);
///
/// \endrst
///
/// Finally, `ModuleDict` provides a lightweight container API, such as allowing
/// iteration over submodules, positional access, adding a new module after
/// construction via `push_back`, as well as joining two `ModuleDict`s via
/// `extend`.

class ModuleDictImpl : public Cloneable<ModuleDictImpl> {
 public:
  using Iterator = OrderedDict<std::string, std::shared_ptr<Module>>::Iterator;
  using ConstIterator =
      OrderedDict<std::string, std::shared_ptr<Module>>::ConstIterator;

  ModuleDictImpl() = default;

  template <typename M>
  explicit ModuleDictImpl(
      const torch::OrderedDict<std::string, std::shared_ptr<M>>& modules) {
    // TODO: check here, I don't think I could use this
    // Also register them
    modules_.reserve(modules.size());
    for (auto pair : modules) {
      insert(pair.key(), pair.value());
    }
  }

  explicit ModuleDictImpl(
      const torch::OrderedDict<std::string, Module>& modules) {
    modules_.reserve(modules.size());
    for (auto pair : modules) {
      insert(pair.key(), pair.value());
    }
  }

  template <typename M>
  explicit ModuleDictImpl(
      const torch::OrderedDict<std::string, ModuleHolder<M>>& modules) {
    modules_.reserve(modules.size());
    for (const auto& pair : modules) {
      insert(pair.key(), pair.value());
    }
  }

  /// `reset()` is empty for `ModuleDict`, since it does not have
  /// parameters of its own.
  void reset() override {}

  /// Special cloning function for `ModuleList` because it does not use
  /// `reset()`.
  std::shared_ptr<Module> clone(
      const optional<Device>& device = nullopt) const override {
    auto clone = std::make_shared<ModuleDictImpl>();
    for (const auto& pair : modules_) {
      clone->insert(pair.key(), pair.value()->clone(device));
    }
    return clone;
  }

  /// Pretty prints the `ModuleDict` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override {
    stream << "torch::nn::ModuleDict";
    // TODO::
    stream << "wait to be impl";
  }

  /// Insert the module along with the key into ModuleDict
  void insert(std::string key, std::shared_ptr<Module> module) {
    modules_.insert(key, std::move(module));
    // TODO: the usage of key here, move or not
    register_module(key, modules_[key]);
  }

  /// Unwraps the contained module of a `ModuleHolder` and adds it to the
  /// `ModuleDict`.
  template <typename M>
  void insert(std::string key, const ModuleHolder<M>& module_holder) {
    insert(key, module_holder.ptr());
  }

  /// Adds a new `Module` to the `ModuleDict` container, moving or copying
  /// it into a `shared_ptr` internally. This method allows passing value types,
  /// and letting the container deal with the boxing.
  template <typename M, typename = torch::detail::enable_if_module_t<M>>
  void insert(std::string key, M&& module) {
    using Type = typename std::remove_reference<M>::type;
    insert(key, std::make_shared<Type>(std::forward<M>(module)));
  }

  /// Remove key from the ModuleDict and return its value, throw exception
  /// if the key is not contained. Please check contains(key) before for a
  /// non-throwing access.
  std::shared_ptr<Module> pop(const std::string& key) {
    auto v = modules_[key];
    modules_.erase(key);
    return v;
  }

  /// Return the keys in the dict
  ::std::vector<std::string> keys() const {
    return modules_.keys();
  }

  /// Return the Values in the dict
  ::std::vector<std::shared_ptr<Module>> values() const {
    return modules_.values();
  }

  /// Return an iterator to the start of ParameterDict
  Iterator begin() {
    return modules_.begin();
  }

  /// Return a const iterator to the start of ParameterDict
  ConstIterator begin() const {
    return modules_.begin();
  }

  /// Return an iterator to the end of ParameterDict
  Iterator end() {
    return modules_.end();
  }

  /// Return a const iterator to the end of ParameterDict
  ConstIterator end() const {
    return modules_.end();
  }

  /// Return the number of items currently stored in the ParameterDict
  size_t size() const noexcept {
    return modules_.size();
  }

  /// Return true if the ParameterDict is empty, otherwise return false
  bool empty() const noexcept {
    return modules_.is_empty();
  }

  /// Returns the value associated with the given `key`. Throws an exception if
  /// no such key is stored in the `ParameterDict`. Check contains(key) before
  /// for a non-throwing way of access
  template <typename T>
  const T& get(const std::string& key) const {
    return *modules_[key]->as<T>();
  }

  /// Returns the value associated with the given `key`. Throws an exception if
  /// no such key is stored in the `ParameterDict`. Check contains(key) before
  /// for a non-throwing way of access
  template <typename T>
  T& get(const std::string& key) {
    return *modules_[key]->as<T>();
  }

  /// Returns the value associated with the given `key`. Throws an exception if
  /// no such key is stored in the `ParameterDict`. Check contains(key) before
  /// for a non-throwing way of access
  std::shared_ptr<Module> operator[](const std::string& key) {
    return modules_[key];
  }

  /// Attempts to return a `std::shared_ptr` whose dynamic type is that of the
  /// underlying module at the given index. Throws an exception if the index is
  /// out of bounds.
  std::shared_ptr<Module> ptr(const std::string& key) const {
    return modules_[key];
  }

  /// Attempts to return a `std::shared_ptr` whose type is the one provided.
  /// Throws an exception if the index is out of bounds or the types do not
  /// match.
  template <typename T>
  std::shared_ptr<T> ptr(const std::string& key) const {
    static_assert(
        torch::detail::is_module<T>::value,
        "Can only call ModuleList::ptr with an nn::Module type");
    return std::dynamic_pointer_cast<T>(modules_[key]);
  }

 private:
  OrderedDict<std::string, std::shared_ptr<Module>> modules_;
};
TORCH_MODULE(ModuleDict);
} // namespace nn
} // namespace torch