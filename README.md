# [CoNEXT '26] A Cross-Country Measurement Study of Inter-cell Interference in Cellular Networks

In this repository, we release the dataset and scripts used in the CoNEXT '26 paper, *A Cross-Country Measurement Study of Inter-cell Interference in Cellular Networks*

<p align="center">
<img src="route.png" width="400"/>
</p>

**Authors**:
[[Zhekun Yu](https://www.linkedin.com/in/zhekun-yu-444a962a8/)]
[[Yufei Feng](https://www.linkedin.com/in/yufei-feng-7b268820b)]
[[Moinak Ghoshal](https://sites.google.com/view/moinak-ghoshal/home)] 
[[Omar Basit](https://scholar.google.com/citations?user=O8YhcToAAAAJ&hl=en)]
[[Imran Khan](https://imranbuet63.github.io)]
[[Z. Jonny Kong](https://www.jonnykong.com)]
[[Sizhe Wang](https://sizhewang.cn)]
[[Phuc Dinh](https://scholar.google.com/citations?user=87M0_7EAAAAJ&hl=en)]
[[Y. Charlie Hu](https://engineering.purdue.edu/~ychu/)]
[[Dimitrios Koutsonikolas](https://ece.northeastern.edu/fac-ece/dkoutsonikolas/)]


## Repository Structure & Usage

| Directory | Contents |
| --- | --- |
| `data/raw_data/` | Origianl XCAL measurement data including RF metrics and LTE-neighbor logs; application downlink logs |
| `data/intermediate_data/` | Intermediate processed data for plotting |
| `scripts/data_cleaning/` | Data preparation scripts |
| `scripts/plotting/` | Plotting scripts |
| `plots/` | Final paper parsed/visualized results as PDFs |


## Data Access

Use the following command to download large dataset files stored with Git LFS:
```bash
git lfs pull
```

## Plotting

Install the Python dependencies with `python -m pip install -r requirements.txt`, then run `python scripts/plotting/plot_all.py` from the repository root to regenerate every figure PDF in `plots/`.  Use `python scripts/plotting/plot_all.py --list` to see the execution order. Or run single script `python scripts/plotting/fig_[*].py` to generate corresponding figure  in paper.


## 2022 raw data parsing

Run `python scripts/data_cleaning/clean_2022.py` to split measurement runs by operator. Then run `python scripts/data_cleaning/parse_2022_lte_neighbors.py` to parse LTE neighboring cell logs. Results are saved in `data/intermediate_data/2022` for plotting.


## 2024 raw data parsing

Run `python scripts/data_cleaning/clean_2024.py`  to parse the XCAL data  into operator pickles, then `python scripts/data_cleaning/build_complete_ue_level_log.py` builds the combined UE-level per cell CSV. Run `python scripts/data_cleaning/parse_2024_lte_neighbors.py` to parse LTE neighboring cell logs. Results are saved in `data/intermediate_data/2024` for plotting.

---

If there are any questions, feel free to contact us  
([yu.zheku@northeastern.edu](mailto:yu.zheku@northeastern.edu)).



