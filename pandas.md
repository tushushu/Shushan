## 一起阅读源代码 - Pandas

最近在利用周末时间阅读一些知名Python项目的源代码，打算写一系列的专题文章来记录阅读源码的心得体会。

### 1. Pandas是做什么的?
Pandas是一个知名的Python library，提供了强大的数据分析工具。数据分析师、算法工程师包括数据工程师都会用到这个library。`import pandas as pd`已经成为了数据科学行业的事实标准，关于用Pandas做数据分析的书也不胜枚举。

### 2. Git指标
* Address: https://github.com/pandas-dev/pandas
* Star: 31.2k
* Used by: 518k
* Contributors: 2450
* Issues: 3.4k
* PR: 145
* Commits: 27,865

截至2021年10月1日，Pandas项目的Github的数据指标如上。不难看出Pandas的受欢迎程度是非常恐怖的，使用者已经达到了518k，Python程序员的基数真的是宇宙无敌。比较尴尬的是Issues竟然都有3.4k，甚至比代码贡献者还多，替开发团队捏一把汗啊。顺便说一句，笔者也是这2450个代码贡献者其中之一，曾修复过`.isin`方法的性能问题。
注：本文阅读的是Pandas 1.2.x的源码。 


### 3. 项目结构
* 根目录Pandas - 放置了每个Python library必备的`setup.py`用于项目打包和发布，由于是大型项目所以`setup.cfg`也是必备的。其* 余的文件包括不限于release note、配置文件、docker文件和shell脚本等等。
* Pandas/LICENCES - 放置了十余个LICENCES文件，大型项目的排面就是不一样。
* Pandas/asv_bench - 顾名思义是用来测试性能的，Airspeed velocity (asv) 是一个在Python包的整个生命周期内对其进行基准测试的工具。用Pandas处理海量数据的时候，没有人希望等待几个小时之后才能得到结果。这个工具用起来很方便，开发完一个git branch之后，用`asv run`命令跑一下Pandas/asv_bench中的测试代码，就可以得到当前branch和master branch之间的性能对比。如果任何方法/函数的性能下降了，Pandas项目组的成员可能会让你把代码改一改再提交，别问我怎么知道的[捂脸]。
* pandas/ci - 就是Continuous integration的意思，现在算是Git项目的标配了吧。CI给code review省了不少麻烦，指望每个写代码的人都自觉做单元测试、坚持写高质量的代码是不现实的。
* pandas/doc - documentation的缩写，里面存放了Pandas文档所需的文字、数据另外还有能够自动把它们整合在一起的`make.py`脚本。
* pandas/flake8 - 主要是存放了一些针对Cython(注意不是Python)的lint规则。Flake8和Pylint是Python最受欢迎的linter，主要是帮助我们写出干净遵循PEP8规范的Python代码，同时也能检查出代码中常见的错误。比如使用了一个不存在的变量，import了一个module却不使用，在循环体外部访问一个被定义在循环体内部的变量等等。
* pandas/pandas - 放在library代码那部分聊。
* pandas/scripts - 一些检查代码质量的脚本和测试脚本，会被pandas/ci目录中的shell脚本调用。不知道为什么不直接放在pandas/ci路径下？[摊手]

### 4. library结构
library代码放在了pandas/pandas目录下面，在Python的世界中，这种library与项目同名的现象还是很普遍的。当然也有例外，比如大名鼎鼎的机器学习library scikit-learn，library代码就放在了scikit-learn/sklearn下面了。  

* pandas/pandas/_config - 这个package主要是存放一些读、写或设置pandas配置的函数或方法。大家最常用的可能就是`set_option`函数，比如修改打印浮点数时所展示的小数位数，修改`pd.DataFrame`所展示的行数或列数等等。
* pandas/pandas/_libs - 这个package可能是Pandas项目里最核心的部分了，里面存放了大量以`.c`、`.h`、`.pyx`、`.pxd`、`.pxi`等后缀结尾的文件。其中`.c`、`.h`是C语言的代码文件，`.pyx`、`.pxd`、`.pxi`是Cython的代码文件。编写这些文件的目的是通过C语言/Cython编写Python的函数扩展，编译为`.so`文件供Python调用，从而提升Python程序的效率。很多人会觉得Python是一门非常慢的语言，不适合用于生产环境，这其实是非常大的误解。大多数情况下性能并不是问题，而很在意性能的场景下，使用C/C++编写Python的扩展即可。在其他知名的Python项目中都出现了C/C++/Cython的身影，比如Tensorflow、Numpy和Scikit-learn等等。那么为什么Python可以调用C/C++编写的代码呢，答案可以很简单也可以很复杂，猜猜Python的解释器是用什么语言写的？[坏笑]
* pandas/pandas/_testing - Pandas的定制化测试工具package，里面有许多测试用的工具函数，比如测试numpy数组相等、产生随机测试用例、生成临时文件、定制化的warning等等。这个package会与`Pytest` library搭配使用，编写出最终的测试代码。
* pandas/pandas/api - 这个package本身没有太多代码，更多的是代码的搬运工。没有几句`def`，大部分都是`from xxx import xxx`这样的代码。主要存放类型判断、扩展类型注册、indexer相关的通用工具函数。
* pandas/pandas/arrays - 依然是代码的搬运工，把所有Array类都import过来放在这个目录下面而已。
* pandas/pandas/compat - 对不同版本的Python以及pyarrow、numpy、pickle等library进行兼容。以pickle为例，pandas的`read_pickle`方法为例，它会先调用当前安装的pickle library中的`pickle.load`方法，如果运行失败，会继续尝试调用`pandas.compat.pickle_compat`方法。虽然Python 3.9的官方文档明确指出pickle library是向后兼容所有版本的，但Pandas 1.3.3的文档里写到'read_pickle is only guaranteed to be backwards compatible to pandas 0.20.3.'，这点需要读者注意。
* pandas/pandas/core - 放到核心代码那部分聊
* pandas/pandas/errors - 各种自定义警告和错误的package，比如`EmptyDataError`、`DtypeWarning`和`DtypeWarning`等等。比起它们的父类如`ValueError`、`Exception`、`KeyError`来说更加明确，让开发者和用户更好理解发生了什么。
* pandas/pandas/io - io package，我们常用的`read_csv`、`to_csv`和`read_sql`等函数都在这里。实现了对不同格式数据的读写、解析和类型转换等功能。
* pandas/pandas/plotting - 将著名的数据可视化library matplotlib的部分功能如hist、scatter、boxplot等集成到Pandas里面，并提供了傻瓜式的api。可以通过`df.hist()`这样简单的方式给每一列都绘制出直方图，如果要是直接使用`matplotlib`可没有这么简单。
* pandas/pandas/tests - `_testing`主要是提供测试工具，而这个package里面放的是实实在在的单元测试脚本。光子目录就有几十个，用Python这种动态类型语言编写的大项目如果不做好单元测试的话，debug的时候会难到怀疑人生。


### 5. 核心代码
通过上面的介绍，相信大家对Pandas项目的结构已经有了初步的了解，接下来我们一起看看核心代码部分。对于Pandas的用户来说，使用频率最高的类就是`DataFrame`，那我们分析一下这个类是怎么实现的。  

#### 5.1 DataFrame类的继承关系
点开文件`pandas/pandas/core/frame.py`发现`DataFrame`类果然在这里。往下一翻我直接“好家伙”，整整10860行。那我就不贴代码了[狗头]。`DataFrame`的父类有两个分别是`OpsMixin`和`NDFrame`。  

其中`OpsMixin`是个非常有趣的类，它把Python的比较操作符（例如__lt__）、逻辑操作符（例如__and__）和二元操作符（例如__add__）等魔术方法进行了定义，并且把这些方法共享的逻辑通过`unpack_zerodim_and_defer`装饰器进行了抽象。同时又为上述三种操作符定义了三个抽象方法`_cmp_method`、`_logical_method`和`_arith_method`，让开发者用自定义的方式而非魔术方法来实现这些操作符。`OpsMixin`原本的代码很多，在这里截取了部分代码如下：
```Python
class OpsMixin:
    # -------------------------------------------------------------
    # Comparisons

    def _cmp_method(self, other, op):
        return NotImplemented

    @unpack_zerodim_and_defer("__lt__")
    def __lt__(self, other):
        return self._cmp_method(other, operator.lt)

    # -------------------------------------------------------------
    # Logical Methods

    def _logical_method(self, other, op):
        return NotImplemented

    @unpack_zerodim_and_defer("__and__")
    def __and__(self, other):
        return self._logical_method(other, operator.and_)

    # -------------------------------------------------------------
    # Arithmetic Methods

    def _arith_method(self, other, op):
        return NotImplemented

    @unpack_zerodim_and_defer("__add__")
    def __add__(self, other):
        return self._arith_method(other, operator.add)
```

而`NDFrame`作为`DataFrame`和另一个Pandas的重要数据结构`Series`（可以简单理解为一维的DataFrame）共同的父类，实现了两者所共用的一些方法，比如`to_csv`、`sort_values`、`copy`和`head`等常用方法。然而这还没有到源头，`NDFrame`还有两个父类`PandasObject`和`IndexingMixin`。  

`IndexingMixin`为`Dataframes`和`Series`实现了`.loc/.iloc/.at/.iat`等方法用来对数据进行索引。其中`.loc/.at`比较类似数组的方式，通过行/列的下标进行索引，而`iloc/.iat`比较类似map的方式，通过key进行查找。


`PandasObject`只是实现了四个方法，代码很少就直接贴上来吧。令人绝望的是它竟然还继承自父类`DirNamesMixin`，我现在看到"Mixin"就头疼。
```Python
class PandasObject(DirNamesMixin):
    """
    Baseclass for various pandas objects.
    """

    # results from calls to methods decorated with cache_readonly get added to _cache
    _cache: dict[str, Any]

    @property
    def _constructor(self):
        """
        Class constructor (for this class it's just `__class__`.
        """
        return type(self)

    def __repr__(self) -> str:
        """
        Return a string representation for a particular object.
        """
        # Should be overwritten by base classes
        return object.__repr__(self)

    def _reset_cache(self, key: str | None = None) -> None:
        """
        Reset cached properties. If ``key`` is passed, only clears that key.
        """
        if not hasattr(self, "_cache"):
            return
        if key is None:
            self._cache.clear()
        else:
            self._cache.pop(key, None)

    def __sizeof__(self) -> int:
        """
        Generates the total memory usage for an object that returns
        either a value or Series of values
        """
        memory_usage = getattr(self, "memory_usage", None)
        if memory_usage:
            mem = memory_usage(deep=True)
            return int(mem if is_scalar(mem) else mem.sum())

        # no memory_usage attribute, so fall back to object's 'sizeof'
        return super().__sizeof__()

```

`DirNamesMixin`重写了`__dir__`魔术方法，调用的时候只会列出部分方法而不是全部，可能是为了便于debug吧。使用了`_accessors`和`_hidden_attrs`两个静态变量可以被所有类的实例共享，避免了重复创建对象所造成的空间浪费，也算是单例模式的一种体现吧。代码不多就贴一下：
```Python
class DirNamesMixin:
    _accessors: set[str] = set()
    _hidden_attrs: frozenset[str] = frozenset()

    def _dir_deletions(self) -> set[str]:
        """
        Delete unwanted __dir__ for this object.
        """
        return self._accessors | self._hidden_attrs

    def _dir_additions(self) -> set[str]:
        """
        Add additional __dir__ for this object.
        """
        return {accessor for accessor in self._accessors if hasattr(self, accessor)}

    def __dir__(self) -> list[str]:
        """
        Provide method name lookup and completion.
        Notes
        -----
        Only provide 'public' methods.
        """
        rv = set(super().__dir__())
        rv = (rv - self._dir_deletions()) | self._dir_additions()
        return sorted(rv)
```  
至此`DataFrame`的继承关系追溯完毕，为了贯彻零拷贝的信条Pandas项目组一层又一层的继承也是拼了。


#### 5.2 高性能的Indexer
上文中提到了`IndexingMixin`为`Dataframes`和`Series`实现了`.loc/.iloc/.at/.iat`等方法用来对数据进行索引，那么这四种索引方式似乎有着一些共同之处，如何进行抽象呢，比如用装饰器？事实证明我还是太肤浅了，Pandas又实现了四个类`_LocIndexer`、 `_iLocIndexer`、 `_AtIndexer`和`_iAtIndexer`来被四个索引方法进行调用。而这四个类也不是源头，它们往上数三代才能找到共同的祖先`pandas/pandas/_libs/indexing.NDFrameIndexerBase`。在<library结构>那部分我们提到过，`pandas/pandas/_libs/`存放了大量C/C++/Cython代码，是Pandas保证高性能的关键。把`NDFrameIndexerBase`的代码实现贴上来：
```Python
cdef class NDFrameIndexerBase:
    """
    A base class for _NDFrameIndexer for fast instantiation and attribute access.
    """
    cdef public:
        str name
        object obj, _ndim

    def __init__(self, name: str, obj):
        self.obj = obj
        self.name = name
        self._ndim = None

    @property
    def ndim(self) -> int:
        # Delay `ndim` instantiation until required as reading it
        # from `obj` isn't entirely cheap.
        ndim = self._ndim
        if ndim is None:
            ndim = self._ndim = self.obj.ndim
            if ndim > 2:
                raise ValueError(
                    "NDFrameIndexer does not support NDFrame objects with ndim > 2"
                )
        return ndim
```
注释写的很清楚：能够快速的创建实例和访问属性。简单解释一下，一般创建一个Python实例的时候会构造一个字典（map）对象`self.__dict__`，并把属性放在这个字典里面。访问属性的时候相当于做了一次字典的查找。`NDFrameIndexerBase`是使用Cython编写的扩展类型，创建实例的时候会使用C struct存储类的属性，与Python的字典相比空间和时间效率都要高。如果很多用户需要在循环体中对`DataFrame`进行索引，那么这个性能改进还是很有必要的。  

#### 5.3 DataFrame的数据结构
Pandas主要是用来分析tabular data的，那么这些数据在`DataFrame`类的内部是以怎样的数据结构存储的呢？让我们抽丝剥茧一探究竟吧。  
首先先看看`DataFrame.__init__`方法是怎样构造一个类的实例的，这个方法的前200行代码都是各种`if-else`分支。因为`DataFrame`支持对多种数据类型进行初始化，比如字典、列表、ndarray等等，所以要写各种各样的逻辑进行适配。跳过这一大段代码终于翻到我们想要的：  
```Python
# ensure correct Manager type according to settings
mgr = mgr_to_mgr(mgr, typ=manager)

NDFrame.__init__(self, mgr)
```
无论传入什么样的数据类型去创建`DataFrame`，最终都会被转换为一个名为`mgr`的对象，并且调用父类`NDFrame.__init__`方法去构造实例。顺藤摸瓜看看`NDFrame`的代码：
```Python
class NDFrame(PandasObject, indexing.IndexingMixin):
    def __init__(
        self,
        data: Manager,
        copy: bool_t = False,
        attrs: Mapping[Hashable, Any] | None = None,
    ):
        # copy kwarg is retained for mypy compat, is not used

        object.__setattr__(self, "_is_copy", None)
        object.__setattr__(self, "_mgr", data)
        object.__setattr__(self, "_item_cache", {})
        if attrs is None:
            attrs = {}
        else:
            attrs = dict(attrs)
        object.__setattr__(self, "_attrs", attrs)
        object.__setattr__(self, "_flags", Flags(self, allows_duplicate_labels=True))
```
注意这里用的是`object.__setattr__`而不是`__setattr__`，这样做的目的是确保Python默认的`__setattr__`函数会被调用，因为有些class的`__setattr__`方法会被自定义。可以看出data（mgr）被放到了`_mgr`属性下面。这个属性是什么类型呢，我试着创建了一个`DataFrame`，打印了一下这个属性发现是`BlockManager`。
```Python
import pandas as pd

data = pd.DataFrame({
    "age": [1, 2, 3],
    "weight": [4, 5, 6],
    "score": [7.8, 8.0, 9.0],
    "name": ['A', 'B', 'C'],
    "gender": ['M', 'F', 'M'],
})

print(data._mgr)
```

```
BlockManager
Items: Index(['age', 'weight', 'score', 'name', 'gender'], dtype='object')
Axis 1: RangeIndex(start=0, stop=3, step=1)
FloatBlock: slice(2, 3, 1), 1 x 3, dtype: float64
IntBlock: slice(0, 2, 1), 2 x 3, dtype: int64
ObjectBlock: slice(3, 5, 1), 2 x 3, dtype: object
```
接下来去点开`BlockManager`的代码，
```Python
    def __init__(
        self,
        blocks: Sequence[Block],
        axes: Sequence[Index],
        do_integrity_check: bool = True,
    ):
        self.axes = [ensure_index(ax) for ax in axes]
        self.blocks: Tuple[Block, ...] = tuple(blocks)
```
所以数据都放到了`BlockManager.blocks`属性下面，我们打印一下刚才生成的`data._mgr.blocks`试试：
```
(FloatBlock: slice(2, 3, 1), 1 x 3, dtype: float64,
 IntBlock: slice(0, 2, 1), 2 x 3, dtype: int64,
 ObjectBlock: slice(3, 5, 1), 2 x 3, dtype: object)
```
可以看出，float、int、str这三种类型的列分别被放到了不同的Block下面。另外顺便吐槽一下，为了实现把不同类型的数据放到不同的block下面，Pandas项目组使用了俄罗斯套娃的编程方式（也就套了七层函数吧），可读性有点一言难尽，有兴趣可以自己去看一下这里就不展开说了。  

我们就看看`IntBlock`里面的代码长啥样吧，它的父类是`NumericBlock`，盲猜这也是`FloatBlock`的父类。而`NumericBlock`的父类又是`Block`，那我们直接看代码：
```Python
class Block(PandasObject):

    def __init__(self, values, placement, ndim: int):
        """
        Parameters
        ----------
        values : np.ndarray or ExtensionArray
        placement : BlockPlacement (or castable)
        ndim : int
            1 for SingleBlockManager/Series, 2 for BlockManager/DataFrame
        """
        # TODO(EA2D): ndim will be unnecessary with 2D EAs
        self.ndim = self._check_ndim(values, ndim)
        self.mgr_locs = placement
        self.values = self._maybe_coerce_values(values)

        if self._validate_ndim and self.ndim and len(self.mgr_locs) != len(self.values):
            raise ValueError(
                f"Wrong number of items passed {len(self.values)}, "
                f"placement implies {len(self.mgr_locs)}"
            )

```
读到这里，终于弄明白`DataFrame`是怎么存放数据的了。比如有`int`型、`float`型、`string`型、`datetime`型4种类型每种4列，那么这16列数据会分别放在对应的`IntBlock`、`FloatBlock`、`ObjectBlock`和`DatetimeBlock`里面，每个`Block`对象的`values`属性会指向一个2D `Numpy`数组来存放这些列数据。这些Block对象会放在BlockManager对象的.blocks属性里，而BlockManager对象会放在DataFrame的_mgr属性里。

### 6. 算法
接下来看看Pandas都实现了哪些算法？

#### 6.1 统计函数
DataFrame的父类NDFrame实现了常用的统计函数，这里把函数的实现方式列一下：
* cummax: np.maximum.accumulate
* cummin: np.minimum.accumulate
* cumsum: np.cumsum
* cumprod: np.cumprod
* var: 根据方差公式的定义计算，利用numpy的向量运算实现
* std: 直接利用DataFrame.var方法计算标准差
* sem: 根据标准误的定义，利用numpy的向量运算实现
* min: np.min
* max: np.max
* mean: 利用np.sum和自定义的count函数计算平均值
* median: np.nanmedian
* skew: 利用np.sum和自定义的count函数计算平均值，再根据skew的定义去计算
* kurt: 与skew类似
Pandas并不是简单的调用numpy函数，它帮助我们做了很多额外的工作。如：
* 适配不同的数据类型，如int8/float16/float32/ExtensionArray/Object/Datetime等
* 缺失值是否需要被跳过
* 适配不同的数据结构 - 1D Series、2D DataFrame和GroupBy对象
* 错误和异常的处理

这里要吐槽一下，Pandas在实现统计学函数如mean/var/skew/kurt的时候，往往是通过.sum()方法进行求和，在数据量比较大的时候会造成浮点数溢出的问题。写均值算法的时候，比如[1, 2, 3]的均值应该采用`1/3 + 2/3 + 3/3 = 2.0`的方式，而不是`(1 + 2 + 3) / 3 = 2.0`。虽然结果相同，但前者能有效避免浮点数溢出的问题。

我试着运行如下代码，果然会溢出：
```Python
>>> import pandas as pd
>>> df = pd.DataFrame([60000.0, 60000.0], dtype="float16")
```

```Python
/Users/username/opt/miniconda3/lib/python3.7/site-packages/numpy/core/_methods.py:47: RuntimeWarning: overflow encountered in reduce
  return umr_sum(a, axis, dtype, out, keepdims, initial, where)

0    inf
dtype: float16
```
针对这个问题，我已经给Pandas社区提了一个issue，如果讨论通过的话我可能会提交个PR修复一下。


#### 6.2 用户自定义函数
很多数据库/数据处理框架都提供自定义函数让用户实现更多的功能，比如Spark、Redshift都会提供`UDF`。Panadas的`DataFrame`提供了一个`apply`方法，能够让用户对数据进行批量操作。举例：
```Python
>>> df = pd.DataFrame({"arr": [1, 2, 3, 4, 5]})
>>> df['arr'].apply(lambda x: x + 1)
```

```Python
0    2
1    3
2    4
3    5
4    6
Name: arr, dtype: int64
```
创建了一个`DataFrame`对象，并将`arr`那一列的元素都加1。这其实用for循环遍历也能实现，但是一般而言用`apply`方法比自己写循环的运行效率要高一些，我们看看apply方法是怎么实现的。

