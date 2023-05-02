# Linkage robot evolution

This repository contains the source code for our paper *Evolution of linkages for prototyping of linkage based robots*.

**Abstract.** Prototyping robotic systems is a time consuming process. Computer aided design, however, might speed up the process significantly. Quality-diversity evolutionary approaches optimise for novelty as well as performance, and can be used to generate a repertoire of diverse designs. This design repertoire could be used as a tool to guide a designer and kick-start the rapid prototyping process. This paper explores this idea in the context of mechanical linkage based robots. These robots can be a good test-bed for rapid prototyping, as they can be modified quickly for swift iterations in design. We compare three evolutionary algorithms for optimising 2D mechanical linkages: 1) a standard evolutionary algorithm, 2) the multi-objective algorithm NSGA-II, and 3) the quality-diversity algorithm MAP-Elites. Some of the found linkages are then realized on a physical hexapod robot through a prototyping process, and tested on two different floors. We find that all the tested approaches, except the standard evolutionary algorithm, are capable of finding mechanical linkages that creates a path similar to a specified desired path. However, the quality-diversity approaches that had the length of the linkage as a behaviour descriptor were the most useful when prototyping. This was due to the quality-diversity approaches having a larger variety of similar designs to choose from, and because the search could be constrained by the behaviour descriptors to make linkages that were viable for construction on our hexapod platform.

<video id="video" width="640" height="360" onclick="play();">
    <source src="videos/AU-1_carpet.MOV" type="video/mov" />
</video>

The article can be found at:

https://ieeexplore.ieee.org/document/10022197

Cite as:



<pre>
E. S. Norstein, K. O. Ellefsen, F. Veenstra, T. Nygaard and K. Glette, "Evolution of linkages for prototyping of linkage based robots," 2022 IEEE Symposium Series on Computational Intelligence (SSCI), Singapore, Singapore, 2022, pp. 1283-1290, doi: 10.1109/SSCI51031.2022.10022197.
</pre>



```
@INPROCEEDINGS{10022197,
  author={Norstein, Emma Stensby and Ellefsen, Kai Olav and Veenstra, Frank and Nygaard, Tønnes and Glette, Kyrre},
  booktitle={2022 IEEE Symposium Series on Computational Intelligence (SSCI)}, 
  title={Evolution of linkages for prototyping of linkage based robots}, 
  year={2022},
  volume={},
  number={},
  pages={1283-1290},
  doi={10.1109/SSCI51031.2022.10022197}}
```
