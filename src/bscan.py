import sys
import os
import shutil
import datetime
import yaml
import requests as rq
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from collections import OrderedDict
import matplotlib.animation as animation
from PIL import Image
import pdb
import time
# local
from data_proc.ers import get_gdm_h5path
from general.path import get_abs_of_file_subdir
import UT
# from scipy.misc import imsave

# global variable used to exit loop when press a key.
exit_loop = False
backward = False
gforward = True
stop = False

cache_dir = os.path.join(r'c:\DATA\GDM', 'Furis')
length = "Length"
side_ = {1: 'L', 2: 'R'}
cnt_len = 639  # 800, 2000, 100000
ms = 3

# dict of defect classes.
HC_keys = [112, 132, 1321, 1322, 212, 422, 232, 2321, 2322]
TC_keys = [211, 411, 421, 431, 471]
RCF_keys = [221, 227, 224, 2223, 2252]
BHC_keys = [135, 235, 236]
BEV_keys = [0]
VSH_keys = [113, 133, 213, 233]
classes = ["HC", "TC", "RCF", "BHC", "BEV", "VSH"]
classes_ = ["RCF", "HC", "TC", "BHC", "BEV", "VSH"]
class_keys_ls = [HC_keys, TC_keys, RCF_keys, BHC_keys, BEV_keys, VSH_keys]
class_dict = {}
for keys, cla in zip(class_keys_ls, classes):
    class_dict.update(dict.fromkeys(keys, cla))

class_ind_dict = dict(enumerate(classes_))
class_ind_dict = dict((v, k) for k, v in class_ind_dict.items())


def get_date_str(y, m, d):
    d = datetime.date(y, m, d)
    date_str = datetime.datetime.strftime(d, "%Y-%m-%d")
    return date_str


def get_drv_match_date(drv_lst, key, date_str):
    return [v for v in drv_lst if v[key] == date_str]


def get_drv_fr_sys_cha(channel, system):
    """get date, runID and versionID from system channel.

    :param channel: channel name
    :param system: system name
    :returns: all requested date, runID and versionID
    :rtype: list of dicts

    """
    url_drv = "http://10.17.96.62:8181/v1/runids"
    pay_load = {"channel": channel, "system": system}
    r_drv = rq.post(url_drv, json=pay_load)
    return r_drv.json()


def get_runids_fr_drv(drv_lst):
    """extract list runIDs from drv list.

    :param drv_lst: input date, runID, versionID list.
    :returns: list of runIDs.
    :rtype: list of number

    """
    return [v['runId'] for v in drv_lst]


def get_meas_bname_fr_runid(runid):
    url_meas_bname = f"http://10.17.96.62:8181/v1/measurementbasename/{runid}"
    r_meas_bname = rq.get(url_meas_bname)
    return r_meas_bname.text


def get_meas_bnames_fr_runids(runids):
    return [get_meas_bname_fr_runid(runid) for runid in runids]


def get_cols_fr_sys_cha(channel, system):
    url_columns = "http://10.17.96.62:8181/v1/columns"
    pay_load = {"channel": channel, "system": system}
    r_columns = rq.post(url_columns, json=pay_load)
    return [v['columnName'] for v in r_columns.json()]


def get_col_data_fr_compdata(channel, system, runid, col_name):
    url_data = "http://10.17.96.62:8181/v1/data"
    post_data = {
        "channel": channel,
        "runId": runid,
        "system": system,
        "valueColumn": col_name
    }
    r_data = rq.post(url_data, json=post_data)
    return r_data.json()


def get_table_data_fr_compdata(channel, system, runid):
    data = pd.DataFrame()
    cols = get_cols_fr_sys_cha(channel, system)
    for col in cols:
        data[col] = get_col_data_fr_compdata(channel, system, runid, col)
    return data


def copy_bscan_hdf_local(runid, system):
    gdm_data_dir = r'\\NLAMFSRFL01.railinfra.local\GDM_RO$\Live\Data'
    if type(runid) is not list:
        runid = [runid]
    else:
        for r_id in runid:
            os.makedirs(cache_dir, exist_ok=True)
            fn = get_gdm_h5path(r_id, system)
            # ipdb.set_trace()
            fnpath = os.path.join(gdm_data_dir, fn)
            fn_new = os.path.basename(fn)
            fnpath_new = os.path.join(cache_dir, fn_new)
            try:
                if os.path.isfile(fnpath_new):
                    # TODO: check also file size
                    raise ValueError
                shutil.copy(fnpath, cache_dir)
                print('{} copied.'.format(fnpath))
            except ValueError:
                print('File {} is already in cache directory.'.format(fn_new))
            except OSError:
                print('{} NOT copied.'.format(fnpath))

            print(f"Path in the GDM for {r_id} is {fnpath}")
            print(f"Path saved local for {r_id} is {fnpath_new}")


def ut_split(ut_echo):
    """this function originates from ut_create_image.py in image_recognition/cnn folder

    :param ut_echo: ut echo pandas DataFrame
    :returns: separated left and right rail side.
    :rtype: dict

    """
    # split by side, side == 1 left?
    cond_l = ut_echo['side'] == 1
    cond_r = ~cond_l
    ut_echo_l = ut_echo[cond_l]
    ut_echo_r = ut_echo[cond_r]
    ut_echo_lr = {"L": ut_echo_l, "R": ut_echo_r}
    return ut_echo_lr


def get_echo_df_fr_runid(runid):
    bname = f'{runid}_Furis.h5'
    f5path = os.path.join(cache_dir, bname)
    # although data maybe copied locally, load_Furis_GDM can also be used.
    df_echo = UT.data.load_Furis_GDM(f5path)
    return df_echo


def get_railh_df_fr_runid(runid):
    bname = f'{runid}_Furis.h5'
    f5path = os.path.join(cache_dir, bname)
    # although data maybe copied locally, load_Furis_GDM can also be used.
    df_railh = UT.data.load_Furis_GDM(f5path, key="Rail height")
    return df_railh


def read_ut_conf(config_ut_fn):
    """this function originates from ut_create_image.py in image_recognition/cnn folder

    :param config_ut_fn:
    :returns:
    :rtype:

    """
    with open(config_ut_fn, 'r') as fh:
        # ipdb.set_trace()
        ut_chans = pd.DataFrame(yaml.load(fh, yaml.CLoader)).T
        ut_chans.columns = ['ch_name', 'offset', 'color']
    return ut_chans


def mask_chans(
    ut_echo,
    ut_chans,
    cnt_start,
    cnt_end,
    railheight=False,
    cnt_type="ExternalCount"
):
    ut_echo_cnt_type = {"ExternalCount": "cnt_ext", "InternalCount": "cnt_int"}
    cnt_t = ut_echo_cnt_type[cnt_type]
    cond_ut = np.logical_and(
        ut_echo[cnt_t] > cnt_start, ut_echo[cnt_t] < cnt_end
    )
    ut_echo_cond = ut_echo[cond_ut]
    if not railheight:
        # classify different channels
        chan_conds = {}
        for cn in ut_chans.index:
            c = (ut_echo_cond['chan_code'] == cn)
            if c.any():
                chan_conds[cn] = c
        return chan_conds, ut_echo_cond
    else:
        # return rail height condition mask.
        return None, ut_echo_cond


def get_ut_config():
    scriptroot = get_abs_of_file_subdir(__file__, '.')
    config_ut_fn = Path(f'{scriptroot}/config_ust02_ut.yaml')
    ut_chans = read_ut_conf(config_ut_fn)
    return ut_chans


def display_echo(
    ax,
    chan_conds,
    ut_chans,
    ut_echo_cond,
    cnt_type="ExternalCount",
    legend=True
):
    ut_echo_cnt_type = {"ExternalCount": "cnt_ext", "InternalCount": "cnt_int"}
    cnt_t = ut_echo_cnt_type[cnt_type]

    for cn in chan_conds:
        c = chan_conds[cn]
        label = ut_chans.loc[cn]['ch_name']
        ax.plot(
            ut_echo_cond[cnt_t][c],
            -ut_echo_cond['depth'][c].astype(float),
            '.',
            color=ut_chans.loc[cn]['color'],
            label=label,
            ms=ms
        )
        if legend:
            ax.legend(loc="upper right")


def display_railh(ax, ut_railh_cond, cnt_type="ExternalCount", legend=True):
    label = "Rail height"
    color = 'k'
    ut_railh_cnt_type = {
        "ExternalCount": "cnt_ext",
        "InternalCount": "cnt_int"
    }
    cnt_t = ut_railh_cnt_type[cnt_type]

    ax.plot(
        ut_railh_cond[cnt_t],
        -ut_railh_cond['depth'].astype(float),
        '.',
        color=color,
        label=label,
        ms=ms
    )
    if legend:
        ax.legend(loc="upper right")


def press(event):
    # print('press', event.key)
    sys.stdout.flush()
    global exit_loop
    global backward
    global forward
    global stop
    if event.key == 'q':
        exit_loop = True
    if event.key == 'b':
        backward = True
        forward = False
    if event.key == 'n':
        forward = True
        backward = False
    if event.key == 'p':
        stop = True


def echo_data_viewer(ut_echo):

    cnt_start = ut_echo['cnt_ext'].iloc[300]
    cnt_len, cnt_shift = 800, 90
    ut_chans = get_ut_config()
    plt.close()
    fig = plt.figure(figsize=(16, 3.5))
    fig.canvas.mpl_connect('key_press_event', press)
    while (1):
        cnt_end = cnt_start + cnt_len
        chan_conds, ut_echo_cond = mask_chans(
            ut_echo, ut_chans, cnt_start, cnt_end
        )
        # plot the depths with different colors
        # canvas = FigureCanvasAgg(fig)
        # fig.patch.set_visible(False)
        ax = fig.gca()
        ax.set_axis_off()
        ax.clear()
        display_echo(ax, chan_conds, ut_chans, ut_echo_cond)
        ax.set_xlim([cnt_start, cnt_end])
        ax.set_ylim([-165, 0])
        # canvas.draw()
        plt.tight_layout()
        fig.canvas.draw()
        # print(f"{cnt_start}-{cnt_end}")
        global backward, forward, stop
        if backward:
            cnt_start -= cnt_shift
        if forward:
            cnt_start += cnt_shift
        if stop:
            plt.pause(0)
        plt.pause(0.001)
        global exit_loop
        if exit_loop:
            plt.close('all')
            break
        # pdb.set_trace()
        # time.sleep(1)


def echo_data_viewer_new(ut_echo):
    # plt.close()
    # fig = plt.figure(figsize=(23, 4))
    # ax = fig.gca()
    global fig
    fig = plt.figure(figsize=(23, 4))
    ax = fig.gca()
    # fig, ax = plt.subplots(figsize=(25, 4))
    ani_ut = AniUT(ut_echo, ax)
    animation.FuncAnimation(
        fig, ani_ut.update, emitter, interval=10, blit=False
    )
    plt.show()


class AniUT:

    def __init__(self, ut_echo, ax):
        self.ax = ax
        self.ut_echo = ut_echo
        self.cnt_len = 1000
        self.ax.set_ylim([-160, 0])

    def update(self):
        cnt_start = self.ut_echo['cnt_ext'].iloc[300]
        cnt_end = cnt_start + self.cnt_len
        ut_chans = get_ut_config()
        chan_conds, ut_echo_cond = mask_chans(
            self.ut_echo, ut_chans, cnt_start, cnt_end
        )
        # plot the depths with different colors
        # canvas = FigureCanvasAgg(fig)
        # fig.patch.set_visible(False)
        self.ax.set_axis_off()
        self.ax.clear()
        plts = []
        for cn in chan_conds:
            c = chan_conds[cn]
            label = ut_chans.loc[cn]['ch_name']
            pl, = self.ax.plot(
                ut_echo_cond['cnt_ext'][c],
                -ut_echo_cond['depth'][c].astype(float),
                '.',
                color=ut_chans.loc[cn]['color'],
                label=label
            )
            plts.append(pl)
        self.ax.set_xlim([cnt_start, cnt_end])
        return plts
        # self.ax.legend(loc="upper right")
        # plts = display_echo(self.ax, chan_conds, ut_chans, ut_echo_cond)

    def init_func(self):
        cnt_start = self.ut_echo['cnt_ext'].iloc[300]
        cnt_end = cnt_start + self.cnt_len
        ut_chans = get_ut_config()
        chan_conds, ut_echo_cond = mask_chans(
            self.ut_echo, ut_chans, cnt_start, cnt_end
        )
        # plot the depths with different colors
        # canvas = FigureCanvasAgg(fig)
        # fig.patch.set_visible(False)
        self.ax.set_axis_off()
        self.ax.clear()
        plts = []
        for cn in chan_conds:
            c = chan_conds[cn]
            label = ut_chans.loc[cn]['ch_name']
            pl, = self.ax.plot(
                ut_echo_cond['cnt_ext'][c],
                -ut_echo_cond['depth'][c].astype(float),
                '.',
                color=ut_chans.loc[cn]['color'],
                label=label
            )
            plts.append(pl)
        self.ax.set_xlim([cnt_start, cnt_end])
        return plts

    def emitter(self):
        cnt_start = self.ut_echo['cnt_ext'].iloc[300]
        cnt_shift = 50
        while (True):
            yield cnt_start
            cnt_start += cnt_shift


def emitter():
    # cnt_start = self.ut_echo['cnt_ext'].iloc[300]
    cnt_start = 4657060.0
    cnt_shift = 50
    while (True):
        yield cnt_start
        cnt_start += cnt_shift


def plot_echo_at_cnt(
    ut_echo,
    cnt,
    side,
    comment="",
    railheight=None,
    cnt_type="ExternalCount",
    **kwargs
):
    cnt_key = "cnts"
    lengths_key = "lengths"
    sides_key = "sides"
    comment_cnt_type = {"ExternalCount": "ExtCnt", "InternalCount": "IntCnt"}
    ut_cnt_type = {"ExternalCount": "cnt_ext", "InternalCount": "cnt_int"}
    if kwargs.get("show_all"):
        cnt_start = ut_echo[ut_cnt_type[cnt_type]].iloc[0]
        cnt_end = ut_echo[ut_cnt_type[cnt_type]].iloc[-1]
    else:
        cnt_start = cnt - cnt_len / 2
        cnt_end = cnt + cnt_len / 2
    alpha_ = .6
    ut_chans = get_ut_config()

    # change axes numbers and figure size if data block also plotted.
    if kwargs.get("plot_data_block"):
        num_axes = 4
        fig_size = (18, 11.5)
    else:
        num_axes = 2
        fig_size = (18, 5.5)
    fig_title = f"{comment}_{comment_cnt_type[cnt_type]}_{cnt}"
    fig, axs = plt.subplots(
        num_axes, 1, num=fig_title, figsize=fig_size, sharex="all"
    )
    axs_ = {'L': axs[0], 'R': axs[1]}
    depth_colo = OrderedDict([("start_depth", 'r'), ("end_depth", 'g')])
    length_colo = {"length": 'r'}
    comment_p = ""
    comment_p_ = ""
    # plot both left and right rails.
    ut_echo_lr = ut_split(ut_echo)
    if railheight is not None:
        ut_railh_lr = ut_split(railheight)
    for k, echo in ut_echo_lr.items():
        chan_conds, ut_echo_cond = mask_chans(
            echo, ut_chans, cnt_start, cnt_end, False, cnt_type
        )
        # mask rail height data.
        if railheight is not None:
            _, ut_railh_cond = mask_chans(
                ut_railh_lr[k], ut_chans, cnt_start, cnt_end, True, cnt_type
            )
            display_railh(axs_[k], ut_railh_cond, cnt_type)

        display_echo(axs_[k], chan_conds, ut_chans, ut_echo_cond, cnt_type)
        # plot depths.
        if side == k:  # show only the side interested.
            for d_k in depth_colo.keys():
                if d_k in kwargs:
                    axs_[k].axhline(
                        y=-kwargs[d_k],
                        ls='-.',
                        linewidth=1,
                        color=depth_colo[d_k],
                        alpha=alpha_
                    )
                    comment_p_ += f"_[{d_k}:{-kwargs[d_k]}]_"
            if comment_p_:
                comment_p = comment + comment_p_
            # plot length.
            len_key = list(length_colo.keys())[0]
            if len_key in kwargs:
                axs_[k].axvline(
                    x=cnt + kwargs[len_key],
                    ls='-.',
                    linewidth=1,
                    color=length_colo[len_key],
                    alpha=alpha_
                )
                len_ = f"_[{len_key}:{kwargs[len_key]}]"
                if not comment_p:
                    comment_p = comment + len_
                else:
                    comment_p += len_
        # multiple length plot.
        if (lengths_key in kwargs) and (cnt_key
                                        in kwargs) and (sides_key in kwargs):
            for l, c, s in zip(
                kwargs[lengths_key], kwargs[cnt_key], kwargs[sides_key]
            ):
                if side_[s] == k:
                    axs_[k].axvline(
                        x=c + l, ls='-.', linewidth=1, color='r', alpha=alpha_
                    )

        if side == k:
            if not comment_p:
                comment_p = comment
            axs_[k].set_title(k + '--' + comment_p)
            color = 'r'
        else:
            axs_[k].set_title(k)
            color = 'g'
        # plot position.
        # for multiple cnts.
        if (cnt_key in kwargs) and (sides_key in kwargs):
            for cnt_, si in zip(kwargs[cnt_key], kwargs[sides_key]):
                if side_[si] == k:
                    colo = 'r'
                else:
                    colo = 'g'
                axs_[k].axvline(x=cnt_, linewidth=1, color=colo)
        else:
            axs_[k].axvline(x=cnt, linewidth=1, color=color)
        axs_[k].set_xlim([cnt_start, cnt_end])
        axs_[k].set_ylim([-175, 5])

    plt.xlabel(cnt_type)
    plt.tight_layout()
    # fig.canvas.draw()
    # save_figures(fig, comment, cnt, cnt_type)
    kwargs.update({
        "fig": fig,
        "cnt_start": cnt_start,
        "cnt_end": cnt_end,
        "fig_title": fig_title
    })
    return kwargs


def save_figures(figure, fig_title, meas_bname_subfolder=True):
    # save figure to suspects and objects folders.
    susp_dir = "../suspects/"
    obj_dir = "../objects/"
    if meas_bname_subfolder:
        sub_folder = fig_title.split('_')[1]
        susp_dir += sub_folder
        obj_dir += sub_folder
    if not os.path.isdir(susp_dir):
        os.mkdir(susp_dir)
    if not os.path.isdir(obj_dir):
        os.mkdir(obj_dir)
    if "Suspect" in fig_title:
        abs_path = os.path.join(susp_dir, f"{fig_title}.jpg")
        if not os.path.isfile(abs_path):
            figure.savefig(abs_path)
            print(fig_title + ".jpg" + " is saved!")
        else:
            print(fig_title + ".jpg" + " already exists!")
    elif "Object" in fig_title:
        abs_path = os.path.join(obj_dir, f"{fig_title}.jpg")
        if not os.path.isfile(abs_path):
            figure.savefig(abs_path)
        else:
            print(fig_title + ".jpg" + "already exists!")
    pass


def plot_echo_susp_at_cnt(
    ut_echo,
    susp_table,
    meas_bname,
    railheight=None,
    cnt_type="ExternalCount",
    **kwargs
):
    all_in_one = False
    plot_data_block = True
    for i in range(susp_table.shape[0]):
        s = susp_table["Side"].iloc[i]
        side = side_[s]
        comment = "Suspect_" + meas_bname + "_" + susp_table["Comment"].iloc[
            i] + "_Cla_" + susp_table["Classification"].astype(str).iloc[
                i] + "_UIC_" + susp_table["UIC"].astype(str).iloc[i]
        # ipdb.set_trace()
        if all_in_one:
            kwargs.update({
                "start_depth": susp_table["StartDepth"].iloc[i],
                "end_depth": susp_table["EndDepth"].iloc[i],
                "length": susp_table["Length"].loc[i],
                "cnts": susp_table[cnt_type],
                "lengths": susp_table[length],
                "sides": susp_table["Side"],
                "show_all": True,
                "plot_data_block": plot_data_block
            })
        else:
            kwargs.update({
                "start_depth": susp_table["StartDepth"].iloc[i],
                "end_depth": susp_table["EndDepth"].iloc[i],
                "length": susp_table["Length"].loc[i],
                "show_all": False,
                "plot_data_block": plot_data_block
            })
        kwargs = plot_echo_at_cnt(
            ut_echo,
            susp_table[cnt_type].iloc[i],
            side,
            comment,
            railheight,
            cnt_type=cnt_type,
            **kwargs
        )
        plot_data_block_at_cnt(
            kwargs["data_block"],
            susp_table[cnt_type].iloc[i],
            side,
            cnt_type=cnt_type,
            **kwargs
        )
        plt.show()
        pdb.set_trace()
        save_figures(kwargs["fig"], kwargs["fig_title"])
        plt.close(kwargs["fig"])
        if all_in_one:
            break


def plot_echo_obj_at_cnt(
    ut_echo, obj_table, meas_bname, railheight=None, cnt_type="ExternalCount"
):
    for i in range(obj_table.shape[0]):
        s = obj_table["Side"].iloc[i]
        side = side_[s]
        comment = "Object_" + meas_bname + "_" + obj_table["Type"].iloc[i]
        plot_echo_at_cnt(
            ut_echo,
            obj_table[cnt_type].iloc[i],
            side,
            comment,
            railheight,
            cnt_type=cnt_type,
            start_depth=obj_table["StartDepth"].iloc[i],
            end_depth=obj_table["EndDepth"].iloc[i],
            length=obj_table["Length"].loc[i]
        )


def print_num_susp_or_obj(table):
    if table.size > 0:
        num_l = sum(table["Side"] == 1)
        num_r = sum(table["Side"] == 2)
        num = table.shape[0]
        assert (num == num_l + num_r)
        print(
            f"{list(table['Operator'].unique())} has found {num_l} suspects on the left side!"
        )
        print(
            f"{list(table['Operator'].unique())} has found {num_r} suspects on the right side!)"
        )
        return num_l, num_r
    else:
        return None, None


def create_df_echo_chan_cols(ut_echo):
    """rearrange ut echo DataFrame with channel name as columns

    :param ut_echo: ut echo DataFrame
    :returns: ut echo DataFrame left and right side of the rail with channel name columns specifying depth.
    :rtype: Pandas DataFrame

    """
    ut_echo_ = ut_echo.copy()
    chan_names = ut_echo['chan_name'].unique()
    for c_n in chan_names:
        c_n_mask = ut_echo['chan_name'] == c_n
        # pdb.set_trace()
        ut_echo_[c_n] = ut_echo[c_n_mask].loc[:, 'depth']

    return ut_echo_, chan_names


def clean_df_echo_chan_cols(ut_echo_cha_cols, chan_names):
    """delete unused columns from the rearranged DataFrame

    :param ut_echo_cha_cols: input DataFrame with channel name as columns.
    :param chan_names: channel names belong to this DataFrame.
    :returns: only data will be used.
    :rtype: Pandas DataFrame.

    """
    cols = np.append(['cnt_int', 'cnt_ext', 'side'], chan_names)
    ut_echo_cha_cols = ut_echo_cha_cols.loc[:, cols]
    return ut_echo_cha_cols


def df_echo_chan_cols_add_railh(ut_echo_cha_cols, ut_railh, chan_names):
    """add rail height column to ut echo DataFrame

    :param ut_echo_cha_cols: input ut echo DataFrame
    :param ut_railh: input railheight DataFrame.
    :returns: ut echo and rail height DataFrame.
    :rtype: DataFrame

    """
    ut_merge = ut_echo_cha_cols.append(ut_railh, sort=False)
    ut_merge.sort_values(by=['cnt_ext'], inplace=True)
    ut_merge.rename(columns={'depth': 'railh'}, inplace=True)
    cols = np.append(['cnt_int', 'cnt_ext', 'side', 'railh'], chan_names)
    ut_merge = ut_merge[cols]
    # pdb.set_trace()
    return ut_merge.reset_index(drop=True)


def plot_all_susps(susp_table, side, axs_img, cnt_type="ExternalCount"):
    """add the suspects position for the whole run.

    :param susp_table: suspects table
    :param cnt_type: type of cnt
    :param side: 1 or 2
    :param axs_img: imshow axis
    :returns: imshow axis with suspects location.
    :rtype: AxesSubplot

    """
    if not susp_table.empty:
        plot_length = True
        alpha_ = .5
        susp_t = susp_table[susp_table['Side'] == side]
        if plot_length:
            key_len = "Length"
            for cnt, l in zip(susp_t[cnt_type], susp_t[key_len]):
                axs_img.axvline(cnt, linewidth=1, color='r', alpha=alpha_)
                axs_img.axvline(
                    cnt + l, ls='-.', linewidth=1, color='r', alpha=alpha_
                )
        else:
            for cnt in susp_t[cnt_type]:
                axs_img.axvline(cnt, linewidth=1, color='r', alpha=alpha_)
    return axs_img


def plot_data_block_at_cnt(
    df_dblock_lr, cnt, side, cnt_type="ExternalCount", **kwargs
):
    """add the suspects position on the channels depth image representation for a region. (cnt_start, cnt_end)

    :param df_dblock_lr: 
    :param cnt: 
    :param side: 1 or 2
    :param cnt_type: type of cnt
    :returns: imshow axis with suspects location.
    :rtype: AxesSubplot

    """
    norm = colors.Normalize(0, 255)
    alpha_ = .5
    len_key = "length"
    fig = kwargs["fig"]
    axs_ = {'L': fig.get_axes()[2], 'R': fig.get_axes()[3]}
    cnt_start = kwargs["cnt_start"]
    cnt_end = kwargs["cnt_end"]
    df_dblock_lr_ = get_data_block_cnt_lr(
        df_dblock_lr, cnt_start, cnt_end, cnt_type
    )
    for k, dblock in df_dblock_lr_.items():
        # pdb.set_trace()
        columns = list(enumerate(dblock.keys()[2:], 1))
        axs_title = f"{k}-{columns}"

        # the masking commented because we only need the interested
        # region to reindex and display at a time.
        # mask = np.logical_and(dblock.index > cnt_start, dblock.index < cnt_end)
        # dblock_ = dblock[mask]

        axs_[k].cla()
        # pdb.set_trace()
        dblock_ = dblock.fillna(255)
        axs_[k].imshow(
            dblock_.iloc[:, 2:].T,
            aspect='auto',
            norm=norm,
            cmap='gray',
            extent=[cnt_start, cnt_end, 10.5, 0.5],
            interpolation='none'
        )
        pdb.set_trace()
        if side == k:
            axs_[k].axvline(cnt, linewidth=1, color='r', alpha=alpha_)
            if len_key in kwargs:
                axs_[k].axvline(
                    cnt + kwargs[len_key],
                    ls='-.',
                    linewidth=1,
                    color='r',
                    alpha=alpha_
                )
        else:
            axs_[k].axvline(cnt, linewidth=1, color='g', alpha=alpha_)
        axs_[k].set_title(axs_title)
        fig.canvas.draw()


def plot_data_block_susp_at_cnt(
    ut_img, susp_table, side, fig, axs_img, cnt_type="ExternalCount"
):
    """add the suspects position on the channels depth image representation for a region.

    :param susp_table: suspects table
    :param cnt_type: type of cnt
    :param side: 1 or 2
    :param axs_img: imshow axis
    :returns: imshow axis with suspects location.
    :rtype: AxesSubplot

    """
    if not susp_table.empty:
        cnt_len = 3000
        plot_length = True
        alpha_ = .5
        susp_t = susp_table[susp_table['Side'] == side]
        norm = colors.Normalize(0, 160)
        columns = list(enumerate(ut_img.keys()[4:], 1))
        axs_img_title = f"{side}-{columns}"
        axs_img.set_title(axs_img_title)
        # fig.colorbar(axs_img, orientation='horizontal', shrink=.15)
        if plot_length:
            key_len = "Length"
            for cnt, l in zip(susp_t[cnt_type], susp_t[key_len]):
                cnt_start = cnt - cnt_len / 2
                cnt_end = cnt + cnt_len / 2
                mask = np.logical_and(
                    ut_img.index > cnt_start, ut_img.index < cnt_end
                )
                ut_img_ = ut_img[mask]
                axs_img.cla()
                axs_img.imshow(
                    ut_img_.iloc[:, 4:].T,
                    aspect='auto',
                    norm=norm,
                    cmap='jet',
                    extent=[cnt_start, cnt_end, 10.5, 0.5]
                )
                axs_img.axvline(cnt, linewidth=1, color='r', alpha=alpha_)
                axs_img.axvline(
                    cnt + l, ls='-.', linewidth=1, color='r', alpha=alpha_
                )
                fig.canvas.draw()
        else:
            for cnt in susp_t[cnt_type]:
                axs_img.axvline(cnt, linewidth=1, color='r', alpha=alpha_)
    return axs_img


def get_data_block_cnt_lr(
    df_data_lr, cnt_start=None, cnt_end=None, cnt_type="ExternalCount"
):
    """make the data block continuously on the cnt according to the cnt type.
    if not values for cnt_start and cnt_end, the whole run will be processed, this may take a lot of memory.

    :param df_data_lr: raw data block left and right in format of Python dict.
    :param cnt_type: type of cnt.
    :returns: left and right data block.
    :rtype: Python dictionary.

    """

    if cnt_type == "ExternalCount":
        cnt_t = "cnt_ext"
    elif cnt_type == "InternalCount":
        cnt_t = "cnt_int"
    else:
        raise ("cnt type needs to be specified correctly!")
    df_dblock_lr = {}
    for s in range(1, 3):
        df_dblock = df_data_lr[side_[s]].groupby(cnt_t).first()
        if all(v is not None for v in [cnt_start, cnt_end]):
            x1, x2 = cnt_start, cnt_end
        else:
            x1, x2 = df_dblock.index[0], df_dblock.index[-1]
        new_index = range(int(x1), int(x2 + 1))
        df_dblock = df_dblock.reindex(new_index).astype(np.float32)
        df_dblock_lr[side_[s]] = df_dblock
    return df_dblock_lr


def ut_echo_plot_2_rgb_array(
    ut_echo, cnt, side, railheight=None, cnt_type="ExternalCount", **kwargs
):
    uic = kwargs.get("UIC")
    # only save the defects in the dict.
    if class_dict.get(uic) is None:
        return kwargs
    if kwargs.get("length") > cnt_len / 4:
        return kwargs
    comment_cnt_type = {"ExternalCount": "ExtCnt", "InternalCount": "IntCnt"}
    ut_cnt_type = {"ExternalCount": "cnt_ext", "InternalCount": "cnt_int"}
    depth_colo = OrderedDict([("start_depth", 'r'), ("end_depth", 'g')])
    length_colo = {"length": 'r'}
    alpha_ = .6
    y_min, y_max = -181, 10
    ut_chans = get_ut_config()
    if kwargs.get("show_all"):
        cnt_start = ut_echo[ut_cnt_type[cnt_type]].iloc[0]
        cnt_end = ut_echo[ut_cnt_type[cnt_type]].iloc[-1]
    else:
        cnt_start = cnt - cnt_len / 2
        cnt_end = cnt + cnt_len / 2
    ut_echo_lr = ut_split(ut_echo)
    if railheight is not None:
        ut_railh_lr = ut_split(railheight)
    echo = ut_echo_lr[side]
    chan_conds, ut_echo_cond = mask_chans(
        echo, ut_chans, cnt_start, cnt_end, False, cnt_type
    )
    fs1, fs2 = cnt_end - cnt_start + 1, y_max - y_min + 1
    fig, axs = plt.subplots(figsize=(fs1 / 100, fs2 / 100))
    time_start = time.time()
    canvas = FigureCanvas(fig)
    # mask rail height data.
    if railheight is not None:
        _, ut_railh_cond = mask_chans(
            ut_railh_lr[side], ut_chans, cnt_start, cnt_end, True, cnt_type
        )
        display_railh(axs, ut_railh_cond, cnt_type, legend=False)
    display_echo(
        axs, chan_conds, ut_chans, ut_echo_cond, cnt_type, legend=False
    )
    # plot depth.
    if False:
        for d_k in depth_colo.keys():
            if d_k in kwargs:
                axs.axhline(
                    y=-kwargs[d_k],
                    ls='-.',
                    linewidth=1,
                    color=depth_colo[d_k],
                    alpha=alpha_
                )
    # plot length.
    if False:
        len_key = list(length_colo.keys())[0]
        if len_key in kwargs:
            axs.vlines(
                x=[cnt, cnt + kwargs[len_key]],
                ymin=y_min,
                ymax=y_max,
                ls='-.',
                linewidth=1,
                color=length_colo[len_key],
                alpha=alpha_
            )

    axs.set_ylim(y_min, y_max)
    axs.set_xlim(cnt_start, cnt_end)
    axs.axis('off')
    fig.tight_layout(pad=0)
    # To remove the huge white borders
    axs.margins(x=0)

    canvas.draw()  # draw the canvas, cache the renderer
    buff = canvas.buffer_rgba()
    image = np.asarray(buff)[:, :, :3]

    img = Image.fromarray(image)
    time_end = time.time()
    print(f"time to convert image -------{time_end-time_start}")
    fig_title = f"{comment_cnt_type[cnt_type]}_{cnt}[{cnt_start}-{cnt_end}].jpg"
    type = kwargs.get('type')
    if kwargs.get(f"{type}_img_path"):
        img_path = kwargs.get(f"{type}_img_path")
        if kwargs.get("meas_bname_subfolder"):
            img_path = os.path.join(img_path, kwargs.get("meas_bname"))
        if not os.path.isdir(img_path):
            os.makedirs(img_path)
        img_abspath = os.path.join(img_path, fig_title)
        img.save(img_abspath)

    # TODO: make images and write annotations to files.
    # make_image()
    xmin, xmax, ymin, ymax = create_bbox(
        cnt, cnt_start, cnt_end, y_min, y_max, **kwargs
    )
    cla = class_dict.get(uic)
    class_id = class_ind_dict.get(cla)
    if class_id == 0:
        write_annotation(
            kwargs.get("wf"), os.path.abspath(img_abspath), str(xmin), str(ymin),
            str(xmax), str(ymax), class_id
        )
    # pdb.set_trace()
    plt.close()
    return kwargs


def plot_2_rgb_arrays(
    ut_echo,
    table,
    meas_name,
    railheight=None,
    cnt_type="ExternalCount",
    **kwargs
):
    kwargs.update({
        "show_all": False,
        "susp_img_path": "../img_src/susp_img",
        "obj_img_path": "../img_src/obj_img",
        "meas_bname": meas_name,
        "meas_bname_subfolder": True,
        "padding": 15,
    })
    a_path = kwargs.get("annotate_txt_path")
    a_name = "bscan_" + kwargs.get("annotate_txt_name") + ".txt"
    f_abs = os.path.join(a_path, a_name)
    wf = open(f_abs, 'a+')
    kwargs.update({"wf": wf})
    for i in range(table.shape[0]):
        if kwargs.get("type") == "susp":
            kwargs.update({"UIC": table["UIC"].iloc[i].astype(int)})
        kwargs.update({
            "start_depth": table["StartDepth"].iloc[i],
            "end_depth": table["EndDepth"].iloc[i],
            "length": table["Length"].loc[i],
        })
        s = table["Side"].iloc[i]
        side = side_[s]
        kwargs = ut_echo_plot_2_rgb_array(
            ut_echo, table[cnt_type].iloc[i], side, railheight, **kwargs
        )
    kwargs.get("wf").close()


def make_image(data, image_path):
    pass


def make_classes(uic):
    pass


def write_annotation(wf, image_path, xmin, ymin, xmax, ymax, class_id):
    annotation = image_path + ' ' + ','.join([
        xmin, ymin, xmax, ymax, str(class_id)
    ])
    wf.write(annotation + '\n')


def create_bbox(cnt, cnt_start, cnt_end, y_min, y_max, **kwargs):
    xmin = cnt - cnt_start
    ymin = y_max + kwargs.get("end_depth")
    xmax = xmin + kwargs.get("length")
    ymax = y_max + kwargs.get("start_depth")
    if kwargs.get("padding"):
        padding = kwargs.get("padding")
        xmin -= padding
        ymin += padding
        xmax += padding
        ymax -= padding
    # recheck the bbox.
    if xmin < 0:
        xmin = 0
    if ymin > y_max - y_min:
        ymin = y_max - y_min
    if xmax > cnt_end - cnt_start:
        xmax = cnt_end - cnt_start
    if ymax < 0:
        ymax = 0
    return xmin, xmax, ymin, ymax
