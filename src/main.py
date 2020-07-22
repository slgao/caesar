# import matplotlib
# matplotlib.use('TkAgg')
import os
import bscan as bsc
import matplotlib.pyplot as plt
# Turn interactive plotting off
# plt.ioff()
import numpy as np
import pdb
import argp

if __name__ == "__main__":
    flags = argp.parser.parse_args()
    anno_txt_name = flags.anno_txt_name
    days = range(1, 28)
    months = range(1, 13)
    get_uic = False
    uic = np.array([])
    kwargs = dict()
    kwargs.update({
        "annotate": True,
        "annotate_txt_name": anno_txt_name,  # "test"
        "annotate_txt_path": "../data/",
    })
    a_path = kwargs.get("annotate_txt_path")
    a_name = "bscan_" + kwargs.get("annotate_txt_name") + ".txt"
    f_abs = os.path.join(a_path, a_name)
    if os.path.exists(f_abs):
        os.remove(f_abs)
    for month in months:
        for day in days:
            channel = "Echo data"
            system = "Furis"
            year = 2020
            key = 'date'
            side = 'L'
            query = True  # test if query set to False.
            show_obj = False
            plot_figures = False
            num_limit = 1000
            # cnt_type = "ExternalCount"
            cnt_type = "ExternalCount"
            plt.close('all')
            # query the database.
            if query:
                drv_lst = bsc.get_drv_fr_sys_cha(channel, system)
                date_str = bsc.get_date_str(year, month, day)
                drv_sel = bsc.get_drv_match_date(drv_lst, key, date_str)
                runid_lst = bsc.get_runids_fr_drv(drv_sel)
                meas_bname_lst = bsc.get_meas_bnames_fr_runids(runid_lst)
                # copy hdf data from database.
                bsc.copy_bscan_hdf_local(runid_lst, system)
            # display the echo data.
            # runid = runid_lst[0]        # 617964
            channel_obj = "Object"
            channel_susp = "Suspect"
            if not query:
                runid_lst = [
                    617964, 617965, 617966, 617967, 617968, 617969, 617970,
                    617971, 617972, 617973, 617974
                ]
                meas_bname_lst = [
                    '20061001us02', '20061002us02', '20061003us02',
                    '20061004us02', '20061005us02', '20061006us02',
                    '20061007us02', '20061008us02', '20061009us02',
                    '20061010us02', '20061011us02'
                ]
            # ind = 9  # [5] with squats, [9] with BEV ? [2] too many figures.
            # runid = runid_lst[ind]  # [3] only 1 susp and obj
            # ipdb.set_trace()
            for ind in range(2, len(runid_lst)):
                if "df_dblock_lr" in locals():
                    del df_dblock_lr
                plt.close('all')
                runid = runid_lst[ind]
                meas_bname = meas_bname_lst[ind]
                # cols_obj = bsc.get_cols_fr_sys_cha(channel_obj, system)
                # cols_susp = bsc.get_cols_fr_sys_cha(channel_susp, system)
                susp_table = bsc.get_table_data_fr_compdata(
                    channel_susp, system, runid
                )
                # get unique values of UIC code.
                if get_uic:
                    uic_ = np.unique(susp_table['UIC'].values)
                    uic = np.append(uic, uic_)
                    uic = np.unique(uic)
                    continue
                obj_table = bsc.get_table_data_fr_compdata(
                    channel_obj, system, runid
                )
                susp_table_l = susp_table[susp_table['Side'] == 1]
                susp_table_r = susp_table[susp_table['Side'] == 2]
                obj_table_l = obj_table[obj_table['Side'] == 1]
                obj_table_r = obj_table[obj_table['Side'] == 2]
                susp_ext_cnts = susp_table_l['ExternalCount']
                obj_ext_cnts = obj_table_l['ExternalCount']
                # col_data = bsc.get_col_data_fr_compdata(channel_susp, system, runid, "Classification")

                df_echo = bsc.get_echo_df_fr_runid(runid)
                df_railh = bsc.get_railh_df_fr_runid(runid)
                df_echo_cha_cols, chan_names = bsc.create_df_echo_chan_cols(
                    df_echo
                )
                df_echo_cha_cols = bsc.clean_df_echo_chan_cols(
                    df_echo_cha_cols, chan_names
                )
                if df_echo is None:
                    continue
                if df_railh is None:
                    continue

                df_echo_cha_railh_cols = bsc.df_echo_chan_cols_add_railh(
                    df_echo_cha_cols, df_railh, chan_names
                )
                df_echo_cha_railh_cols = df_echo_cha_railh_cols.reset_index(
                    drop=True
                )

                df_echo_cha_railh_cols_lr = bsc.ut_split(
                    df_echo_cha_railh_cols
                )
                # pdb.set_trace()
                # df_dblock_lr = bsc.get_data_block_cnt_lr(
                #     df_echo_cha_railh_cols_lr
                # )

                # in order to save memory use not reindexed DataFrame first.
                kwargs.update({"data_block": df_echo_cha_railh_cols_lr})

                fig_img, axs_img = plt.subplots(
                    1,
                    1,
                    num=f"all_channels_depth-{meas_bname}",
                    figsize=(18, 5.5)
                )
                kwargs.update({
                    "susp_table": susp_table,
                    "obj_table": obj_table
                })
                # norm = colors.Normalize(0, 160)
                # im = axs_img.imshow(
                #     df_imshow.iloc[:, 4:].T,
                #     aspect='auto',
                #     norm=norm,
                #     cmap='jet',
                #     extent=[x1, x2, 10.5, 0.5]
                # )
                # columns = list(enumerate(df_imshow.keys()[4:], 1))
                # axs_img_title = f"{side}-{columns}"
                # axs_img.set_title(axs_img_title)
                # fig_img.colorbar(im, orientation='horizontal', shrink=.15)
                # plt.tight_layout()
                # pdb.set_trace()

                df_echo_lr = bsc.ut_split(df_echo)
                # plot some data points. For example, left side.
                # side 1 left, side 2 right.
                df_echo_l = df_echo_lr[side]
                num_l, num_r = bsc.print_num_susp_or_obj(susp_table)
                if num_l is not None:
                    num = num_l + num_r
                    if num < num_limit:
                        if True:
                            if susp_table.size > 0:
                                # bsc.plot_2_rgb_arrays(
                                #     df_echo, susp_table, meas_bname, df_railh,
                                #     cnt_type, **kwargs
                                # )
                                type = kwargs["type"] = "susp"  # obj
                                bsc.plot_2_rgb_arrays(
                                    df_echo, kwargs[f"{type}_table"],
                                    meas_bname, df_railh, cnt_type, **kwargs
                                )
                                # bsc.plot_echo_susp_at_cnt(
                                #     df_echo, susp_table, meas_bname, df_railh,
                                #     cnt_type, **kwargs
                                # )
                                # pdb.set_trace()

                                # only side 1 at the moment.
                                # bsc.plot_all_susps(susp_table, 1, axs_img)

                                # plot data block on range (cnt_start, cnt_end).
                                # ut_img = df_dblock_lr[side]
                                # bsc.plot_data_block_susp_at_cnt(
                                #     ut_img, susp_table, 1, fig_img, axs_img
                                # )
                                # pdb.set_trace()
                            if show_obj:
                                if obj_table.size > 0:
                                    bsc.plot_echo_obj_at_cnt(
                                        df_echo, obj_table, meas_bname,
                                        df_railh, cnt_type
                                    )
                            if plot_figures:
                                plt.show()

                # bsc.echo_data_viewer(df_echo_l)

                # animation dose not work yet.
                # bsc.echo_data_viewer_new(df_echo_l)
