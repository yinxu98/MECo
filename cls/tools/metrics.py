import os
import re

from pandas import DataFrame

folder_workdir = 'rsscn'


def get_taskname(ls_line):
    line_taskname = [line for line in ls_line if 'work_dirs' in line][0]
    result = re.match('.*work_dirs/(.*)\.pth', line_taskname)
    taskname = result.group(1)
    taskname = taskname.replace('_gao', ',gao').replace('_open',
                                                        ',open').replace(
                                                            '_rss', ',rss')
    taskname = taskname.replace('_pretrain/', ',').replace('/', ',')
    taskname = taskname.replace(',01,', ',0.1,').replace(',02,',
                                                         ',0.2,').replace(
                                                             ',05,', ',0.5,')
    method, dataset, epoch = taskname.split(',')
    dataset = dict(opensarship='ship', gaofen4plane='plane',
                   rsscn='rsscn')[dataset]
    return method, dataset, epoch


def get_percentage(ls_line):
    line_lr = [line for line in ls_line if 'percentage' in line][0]
    result = re.match('.*percentage=(.*),', line_lr)
    percentage = result.group(1)
    return percentage


def get_lr(ls_line):
    line_lr = [line for line in ls_line if 'optimizer' in line][0]
    result = re.match('.*lr=(.*), mo.*', line_lr)
    lr = result.group(1)
    return lr


def get_acc(ls_line):
    ls_line = [line for line in ls_line if line.startswith('Test')]
    # ls_line = ls_line[:200]

    dc_best = dict(acc=(0, 0), f1=(0, 0))

    for line in ls_line:
        result = re.match('.*acc (.*) f1 (.*) conf_mat', line)
        acc, f1 = result.group(1), result.group(2)
        acc = 100 * float(acc)
        if acc > dc_best['acc'][0]:
            dc_best['acc'] = (acc, f1)
        if f1 != 'nan':
            f1 = 100 * float(f1)
            if f1 > dc_best['f1'][1]:
                dc_best['f1'] = (acc, f1)

    return dc_best


if __name__ == '__main__':
    ls_result = list()
    ls_folder_task = [
        folder for folder in os.listdir(folder_workdir)
        if folder.startswith('classifier')
    ]

    for folder_task in ls_folder_task:
        ls_log = os.listdir(os.path.join(folder_workdir, folder_task))

        taskmode = folder_task.split('_')[-1]

        for log in ls_log:
            file_log = os.path.join(folder_workdir, folder_task, log)

            with open(file_log, 'r') as fin:
                ls_line = fin.readlines()

            method, dataset, epoch = get_taskname(ls_line)
            lr = get_lr(ls_line)
            percentage = get_percentage(ls_line)
            dc_best = get_acc(ls_line)

            ls_result.append([
                dataset,
                method,
                percentage,
                lr,
                taskmode,
                epoch,
                f'{dc_best["acc"][0]:.2f}',
                log,
            ])

    columns = [
        'dataset',
        'method',
        'percentage',
        'lr',
        'taskmode',
        'epoch',
        'acc',
        'log',
    ]
    file_result = os.path.join(folder_workdir, 'result.xlsx')
    df_result = DataFrame(ls_result, columns=columns)
    df_result = df_result.sort_values(by=columns)
    df_result.to_excel(file_result, index=False, header=False)
