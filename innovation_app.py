import csv
import os
import sys
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QMessageBox, QDialog, QFileDialog, QLabel, QComboBox, QVBoxLayout, QRadioButton, \
    QPushButton, QWidget
import pandas as pd
import math
from PyQt5 import QtCore
from PyQt5.uic.properties import QtCore
from scipy.optimize import Bounds, minimize
from sklearn.linear_model import LinearRegression
import numpy as np
from PyQt5.QtWidgets import QTableWidgetItem
from PyQt5.QtWidgets import QTableWidget
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq, irfft
import copy

from sklearn.metrics import r2_score

url_1 = 'D:\\STUDY\\симонов\\для диплома\\файлы_данные\\УровеньИДО.csv'

def growth_rates(ch):
    base_value = ch[0]
    chain_growth_rates = [(ch[i] / ch[i-1]) for i in range(1, len(ch))]
    base_growth_rates = [(value / base_value) for value in ch]
    chain_absolute_growth = [ch[i] - ch[i-1] for i in range(1, len(ch))]
    base_absolute_growth=[(ch[i]- ch[0]) for i in range(0, len(ch))]
    chain_growth_rate_of_growth = [(chain_growth_rates[i] - chain_growth_rates[i-1]/chain_growth_rates[i-1])
                                   for i in range(0, len(chain_growth_rates))]
    base_growth_rate_of_growth = [(base_growth_rates[i]- base_growth_rates[0]/base_growth_rates[0])
                                  for i in range(0, len(base_growth_rates))]
    return (chain_growth_rates, base_growth_rates, chain_absolute_growth,base_absolute_growth,
            chain_growth_rate_of_growth,base_growth_rate_of_growth)


def format_growth_rates(chain_growth, base_growth, chain_absolute_growth, base_absolute_growth, chain_growthofgrowth, base_growthofgrowth):
    chain_growth_format = [float('%.2f' % elem) for elem in chain_growth]
    base_growth_format = [float('%.2f' % elem) for elem in base_growth]
    chain_absolute_format = [float('%.2f' % elem) for elem in chain_absolute_growth]
    base_absolute_format = [float('%.2f' % elem) for elem in base_absolute_growth]
    chain_growthofgrowth_format=[float('%.2f' % elem) for elem in chain_growthofgrowth]
    base_growthofgrowth_format=[float('%.2f' % elem) for elem in base_growthofgrowth]
    chain_growth_format.insert(0,np.nan)
    chain_absolute_format.insert(0,np.nan)
    chain_growthofgrowth_format.insert(0,np.nan)
    return chain_growth_format, base_growth_format, chain_absolute_format, base_absolute_format, chain_growthofgrowth_format, base_growthofgrowth_format


def create_dataframes(areas, years):
    dataframes = {}
    for i, area in enumerate(areas):
        chain_growth, base_growth, chain_absolute_growth, base_absolute_growth, chain_growthofgrowth, base_growthofgrowth = growth_rates(
            area)
        chain_growth, base_growth, chain_absolute_growth, base_absolute_growth, chain_growthofgrowth, base_growthofgrowth = format_growth_rates(
            chain_growth, base_growth, chain_absolute_growth, base_absolute_growth, chain_growthofgrowth,
            base_growthofgrowth)

        df_chain_growth = pd.DataFrame({
            "Год": years,
            "Цепной темп роста": chain_growth,
            "Базисный тесп роста": base_growth
        })

        df_growth_of_growth = pd.DataFrame({
            "Год": years,
            "Цепной темп прироста": chain_growthofgrowth,
            "Базисный темп прироста": base_growthofgrowth
        })

        df_absolute_growth = pd.DataFrame({
            "Год": years,
            "Цепной абсолютный прирост": chain_absolute_growth,
            "Базисный абсолютный прирост": base_absolute_growth
        })
        dataframes[f"Область {i + 1}"] = {
            "Темпы роста": df_chain_growth,
            "Темпы прироста": df_growth_of_growth,
            "Абсолютные приросты": df_absolute_growth
        }
    return dataframes

def rolling_means(headers,areas, window_size):
        plt.rc('font', family='Times New Roman')
        all_dfs = []
        combined_df = pd.DataFrame()
        rolling_mean_final=[]
        for i, area in enumerate(areas):
            years = np.arange(len(area))
            nan_values = [np.nan]
            if window_size == 5:
                rolling_mean = pd.Series(area).rolling(window=window_size).mean().iloc[
                               window_size - window_size // 2 - 1:].values
                rolling_mean_final = list(rolling_mean) + 2 * nan_values
            elif window_size == 3:
                rolling_mean = pd.Series(area).rolling(window=window_size).mean().iloc[
                               window_size - window_size // 2:].values
                rolling_mean_final = nan_values + list(rolling_mean) + nan_values

            rolling_mean_final = np.round(rolling_mean_final, 3)
            print(rolling_mean_final)
            df = pd.DataFrame({
                'Год': years,
                f'{headers[i]}': rolling_mean_final
            })
            all_dfs.append(df)

            if combined_df.empty:
                combined_df = df
            else:
                combined_df = pd.merge(combined_df, df, on='Год', how='outer')

            print(df)
            # plot_rolling_means_single(rolling_mean_final, years, area, i)

        print("Общий DataFrame:")
        print(combined_df)
        # plot_rolling_means_combined(areas, combined_df, window_size)
        return combined_df,all_dfs

class optimizer_f:
    def __init__(self,y):
        self.work_array=y
    def calculate_f_otrkl(self,x):
        model_a0 = x[0]
        model_ampl = x[1]
        model_freq = x[2]
        model_phase = x[3]
        rav0 = []
        t1 = np.arange(0, len(self.work_array))
        for t in t1:
            x = model_a0 + model_ampl * np.sin((model_freq * t) - (model_phase))
            rav0.append(x)
        rre_sp = []
        sum_sq = 0
        for j in range(0, len(rav0)):
            rre = (self.work_array[j] - rav0[j]) * (self.work_array[j] - rav0[j]) # разница - рассчитанные значения по урав
            rre_sp.append(rre)
            sum_sq += rre
        return sum_sq
    ############# start оптимизируем для минимизации calculate_f_otrkl ###########################
    def opt(self,x,raz):
        max_ampl=raz/2 # raz - размах в разнице
        x_start = x
        bounds = Bounds([0, 0, 0, 0],  # [min x0, min x1]
                                [0, max_ampl,12,12])
        result = minimize(self.calculate_f_otrkl, x_start, method='trust-constr')        #trust-constr — поиск локального минимума в доверительной области.
        return result.x
def nach_dan(file_name):
    dataframe = pd.read_csv(file_name, delimiter=',')
    areas = []
    headers = dataframe.columns[1:].tolist()
    years = dataframe.iloc[:, 0].tolist()
    for i in range(1, len(dataframe.columns)):
        areas.append(dataframe.iloc[:, i].tolist())
    return areas,years,headers
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi("D:\\STUDY\\симонов\\для диплома\\для приложения\\main_prilozh.ui", self)
        self.button_begin.clicked.connect(self.toggle_text)
        self.button_end.clicked.connect(self.close)
        self.setFixedSize(self.size())

    def open_sub_window(self):
        self.sub_window = SubWindow(self)
        self.sub_window.show()

    def toggle_text(self):
        if self.button_begin.text() == "Начать работу":
            self.open_sub_window()

class SpravkaWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(SpravkaWindow, self).__init__(parent)
        uic.loadUi("D:\\STUDY\\симонов\\для диплома\\для приложения\\spravka.ui", self)
        self.spravka_close=self.findChild(QtWidgets.QPushButton,'spravka_close')
        self.spravka_close.clicked.connect(self.close)

class TheorIndWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(TheorIndWindow, self).__init__(parent)
        uic.loadUi("D:\\STUDY\\симонов\\для диплома\\для приложения\\indicator.ui", self)
        self.pushButton=self.findChild(QtWidgets.QPushButton,'pushButton')
        self.pushButton.clicked.connect(self.close)
class TheoryRollMeans(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(TheoryRollMeans, self).__init__(parent)
        uic.loadUi("D:\\STUDY\\симонов\\для диплома\\для приложения\\information_rollmeans.ui", self)
        self.pushButton=self.findChild(QtWidgets.QPushButton,'pushButton')
        self.pushButton.clicked.connect(self.close)

class TheoryFurye(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(TheoryFurye, self).__init__(parent)
        uic.loadUi("D:\\STUDY\\симонов\\для диплома\\для приложения\\inform_furye.ui", self)
        self.pushButton=self.findChild(QtWidgets.QPushButton,'pushButton')
        self.pushButton.clicked.connect(self.close)

class TheoryOptimFurye(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(TheoryOptimFurye, self).__init__(parent)
        uic.loadUi("D:\\STUDY\\симонов\\для диплома\\для приложения\\inform_optim.ui", self)
        self.pushButton=self.findChild(QtWidgets.QPushButton,'pushButton')
        self.pushButton.clicked.connect(self.close)


class SubWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(SubWindow, self).__init__(parent)
        uic.loadUi("D:\\STUDY\\симонов\\для диплома\\для приложения\\sub_window_new.ui", self)
        # self.button_analysis.clicked.connect(self.text_on_but_analysis())
        self.init_ui()

    def init_ui(self):
        self.tabWidget = self.findChild(QtWidgets.QTabWidget, 'tabWidget')
        # Настройка первой вкладки
        self.tabWidget.setCurrentIndex(0)
        self.button_load = self.findChild(QtWidgets.QPushButton, 'button_load')
        self.button_load.clicked.connect(self.open_csv_file)
        self.tableWidget = self.findChild(QtWidgets.QTableWidget, 'tableWidget')
        self.for_ch_rows_2 = self.findChild(QtWidgets.QLabel, 'for_ch_rows_2')
        self.for_ch_col_2 = self.findChild(QtWidgets.QLabel, 'for_ch_col_2')
        self.for_name_file = self.findChild(QtWidgets.QLabel, 'for_name_file')
        self.label = self.findChild(QtWidgets.QLabel, 'label')
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setGeometry(self.tableWidget.geometry())
        self.label.raise_()
        self.show_instruction()
        self.button_analysis = self.findChild(QtWidgets.QPushButton, 'button_analysis')
        self.button_analysis.clicked.connect(self.show_analysis_tab)
        self.button_close = self.findChild(QtWidgets.QPushButton, 'button_close')
        self.button_close.clicked.connect(self.close)
        self.spravka=self.findChild(QtWidgets.QPushButton,'spravka')
        self.spravka.clicked.connect(self.window_spravka_open)
        self.button_check=self.findChild(QtWidgets.QPushButton,'button_check')
        # self.button_check.clicked.connect(self.check_data)


        # Настройка второй вкладки
        self.comboBox = self.findChild(QComboBox, 'comboBox')
        self.comboBox.currentIndexChanged.connect(self.handle_combobox_change)
        self.but_rost=self.findChild(QtWidgets.QRadioButton,'but_rost')
        self.but_prirost = self.findChild(QtWidgets.QRadioButton, 'but_prirost')
        self.but_absrost = self.findChild(QtWidgets.QRadioButton, 'but_absrost')
        self.table_for_indinam=self.findChild(QtWidgets.QTableWidget, 'table_for_indinam')
        self.button_ras=self.findChild(QtWidgets.QPushButton,'button_ras')
        self.button_ras.clicked.connect(self.res_tabl_ind_dinam)
        self.button_graph = self.findChild(QtWidgets.QPushButton, 'button_graph')
        self.button_graph.clicked.connect(self.open_graph_window)
        self.but_keep_table = self.findChild(QtWidgets.QPushButton, 'but_keep_table')
        self.but_keep_table.clicked.connect(self.keep_as_table)
        self.but_next_1=self.findChild(QtWidgets.QPushButton,'but_next_1')
        self.but_next_1.clicked.connect(self.next_vkl)
        self.back_on1=self.findChild(QtWidgets.QPushButton,'back_on1')
        self.back_on1.clicked.connect(self.back_vkl)
        self.crash1=self.findChild(QtWidgets.QPushButton,'crash1')
        self.crash1.clicked.connect(self.close)
        self.next_but_2=self.findChild(QtWidgets.QPushButton,'next_but_2')
        self.next_but_2.clicked.connect(self.next_vkl)
        self.back_2 = self.findChild(QtWidgets.QPushButton, 'back_2')
        self.back_2.clicked.connect(self.back_vkl)
        self.crash2 = self.findChild(QtWidgets.QPushButton, 'crash2')
        self.crash2.clicked.connect(self.close)
        self.inf_formul=self.findChild(QtWidgets.QPushButton,'inf_formul')
        self.inf_formul.clicked.connect(self.open_inform_indicators)
        # Настройка третьей вкладки

        self.rbut_3 = self.findChild(QtWidgets.QRadioButton, 'rbut_3')
        self.rbut_5 = self.findChild(QtWidgets.QRadioButton, 'rbut_5')
        self.table_roll = self.findChild(QtWidgets.QTableWidget, 'table_roll')
        self.button_ras_roll = self.findChild(QtWidgets.QPushButton, 'button_ras_roll')
        self.button_ras_roll.clicked.connect(self.load_rolling_means)
        self.but_graph_rol = self.findChild(QtWidgets.QPushButton, 'but_graph_rol')
        self.but_graph_rol.clicked.connect(self.open_window_graph_rol)
        self.but_saveroll=self.findChild(QtWidgets.QPushButton,'but_saveroll')
        self.but_saveroll.clicked.connect(self.keep_as_table_roll)
        self.toolButton=self.findChild(QtWidgets.QToolButton,'toolButton')
        self.toolButton.clicked.connect(self.open_inform_rollmeans)


        # Настройка четвертой вкладки
        self.graph_trend=self.findChild(QtWidgets.QGraphicsView, 'graph_trend')
        self.table_trend = self.findChild(QtWidgets.QTableWidget, 'table_trend')
        self.but_res_trend=self.findChild(QtWidgets.QPushButton, 'but_res_trend')
        self.but_graph_trend=self.findChild(QtWidgets.QPushButton, 'but_graph_trend')
        self.but_graph_trend.clicked.connect(self.graph_for_trend)
        self.but_res_trend.clicked.connect(self.show_trend_results)
        self.but_keep_trend=self.findChild(QtWidgets.QPushButton, 'but_keep_trend')
        self.but_keep_trend.clicked.connect(self.keep_graph)
        self.comboBox_for_trend = self.findChild(QComboBox, 'comboBox_for_trend')
        self.open_single_trend=self.findChild(QtWidgets.QPushButton, 'open_single_trend')
        self.open_single_trend.clicked.connect(self.open_single_trend_window)
        self.but_contin_trend=self.findChild(QtWidgets.QPushButton, 'but_contin_trend')
        self.but_contin_trend.clicked.connect(self.next_vkl)
        self.back3 = self.findChild(QtWidgets.QPushButton, 'back3')
        self.back3.clicked.connect(self.back_vkl)
        self.crash3 = self.findChild(QtWidgets.QPushButton, 'crash3')
        self.crash3.clicked.connect(self.close)

        # Настройка пятой вкладки
        self.graph_furye=self.findChild(QtWidgets.QGraphicsView, 'graph_furye')
        self.but_graph_furye = self.findChild(QtWidgets.QPushButton, 'but_graph_furye')
        self.inf_furye=self.findChild(QtWidgets.QPushButton, 'inf_furye')
        self.comboBox_for_furye=self.findChild(QComboBox, 'comboBox_for_furye')
        self.but_graph_furye.clicked.connect(self.plot_selected_furye)
        self.but_sled=self.findChild(QtWidgets.QPushButton, 'but_sled')
        self.but_sled.clicked.connect(self.next_vkl)
        self.but_keep_furye=self.findChild(QtWidgets.QPushButton,'but_keep_furye')
        self.but_keep_furye.clicked.connect(self.keep_graph)
        self.back4 = self.findChild(QtWidgets.QPushButton, 'back4')
        self.back4.clicked.connect(self.back_vkl)
        self.crash4 = self.findChild(QtWidgets.QPushButton, 'crash4')
        self.crash4.clicked.connect(self.close)
        self.inf_furye=self.findChild(QtWidgets.QPushButton,'inf_furye')
        self.inf_furye.clicked.connect(self.open_inf_furye)

        # Настройка шестой вкладки
        self.table_for_rasopt=self.findChild(QtWidgets.QTableWidget, 'table_for_rasopt')
        self.comboBox_for_opt=self.findChild(QComboBox, 'comboBox_for_opt')
        self.but_ras_opt=self.findChild(QtWidgets.QPushButton, 'but_ras_opt')
        self.but_graph_opt=self.findChild(QtWidgets.QPushButton, 'but_graph_opt')
        self.graph_for_opt=self.findChild(QtWidgets.QGraphicsView, 'graph_for_opt')

        self.comboBox_for_opt.addItems(["оптимизировать по 1 колебанию", "оптимизировать по 2 колебаниям"])
        self.but_ras_opt.clicked.connect(self.perform_optimization)
        self.but_graph_opt.clicked.connect(self.plot_optimization_results)
        self.but_sing_opt=self.findChild(QtWidgets.QPushButton, 'but_sing_opt')
        self.but_sing_opt = self.findChild(QtWidgets.QPushButton, 'but_sing_opt')
        self.but_sing_opt.clicked.connect(self.plot_optimization_opt)
        self.comboBox_single_opt = self.findChild(QComboBox, 'comboBox_single_opt')

        self.back5 = self.findChild(QtWidgets.QPushButton, 'back5')
        self.back5.clicked.connect(self.back_vkl)
        self.crash5 = self.findChild(QtWidgets.QPushButton, 'crash5')
        self.crash5.clicked.connect(self.close)
        self.but_inf_opt=self.findChild(QtWidgets.QPushButton,'but_inf_opt')
        self.but_inf_opt.clicked.connect(self.open_inf_optim)

    def window_spravka_open(self):
            self.spr_window = SpravkaWindow(self)
            self.spr_window.show()

    def open_inform_indicators(self):
        self.inf_ind_window = TheorIndWindow(self)
        self.inf_ind_window.show()

    def open_inform_rollmeans(self):
        self.inf_roll_window = TheoryRollMeans(self)
        self.inf_roll_window.show()

    def open_inf_furye(self):
        self.inf_furye_window = TheoryFurye(self)
        self.inf_furye_window.show()
    def open_inf_optim(self):
        self.inf_opt_window = TheoryOptimFurye(self)
        self.inf_opt_window.show()

    def next_vkl(self):
        self.comboBox_for_furye.addItems(self.df.columns[1:])
        current_index = self.tabWidget.currentIndex()
        total_tabs = self.tabWidget.count()
        next_index = (current_index + 1) % total_tabs
        self.tabWidget.setCurrentIndex(next_index)
    def back_vkl(self):
        self.comboBox_for_furye.addItems(self.df.columns[1:])
        current_index = self.tabWidget.currentIndex()
        total_tabs = self.tabWidget.count()
        next_index = (current_index - 1) % total_tabs
        self.tabWidget.setCurrentIndex(next_index)

    def keep_graph(self):
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Graph", "", "PNG Files (*.png);;All Files (*)",
                                                       options=options)
            if file_name:
                self.save_graph_data(file_name)
                self.show_message("График сохранен!")

    def save_graph_data(self, file_name):
        self.fig3.savefig(file_name)

    def show_message(self, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText(message)
        msg_box.setWindowTitle("Сообщение")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()


    def perform_optimization(self):
        selected_option = self.comboBox_for_opt.currentText()
        selected_column = self.comboBox_for_furye.currentText()
        t = np.arange(0,23)
        fourier_equations = []
        r2_all = []

        for column in self.df.columns:
            if column != 'Years':
                area = self.df[column].values
                zn = self.raschet_znach_trenda([area], t)
                amplitudes, freqs, phases, N, znach_razn, i, _ = self.furye(zn, [area], column, selected_column)

                if selected_option == "оптимизировать по 1 колебанию":
                    _, sum_trend_i_optznach, fourier_equation, r2, _,_,_,_,_,_ = self.opt_sinus_1(amplitudes, freqs, phases,
                                                                                         znach_razn,
                                                                                         N, i, area, [area])
                elif selected_option == "оптимизировать по 2 колебаниям":
                    _, sum_trend_i_optznach, fourier_equation, r2 = self.opt_sinus_2(amplitudes, freqs, phases,
                                                                                         znach_razn,
                                                                                         N, i, area, [area])
                fourier_equations.append(fourier_equation)
                r2_all.append(r2)
        print(r2_all)
        self.table_for_rasopt.setRowCount(len(fourier_equations))
        self.table_for_rasopt.setColumnCount(3)
        headers=['Область','Уравнение Фурье','Коэффициент детерминации']
        self.set_header_with_wrapping(self.table_for_rasopt,headers)
        # self.table_for_rasopt.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

        for row, equation, r2_one in zip(range(len(fourier_equations)), fourier_equations, r2_all):
            self.table_for_rasopt.setItem(row, 0, QtWidgets.QTableWidgetItem(f"Область {row+1}"))
            self.table_for_rasopt.setItem(row, 1, QtWidgets.QTableWidgetItem(equation))
            self.table_for_rasopt.setItem(row, 2, QtWidgets.QTableWidgetItem(str(r2_one)))

    def opt_sinus_1(self, amplitudes, freqs, phases, y, N, num_obl, area, areas):
        # Реализация первой оптимизации
        model_a0 = 0
        max_amp_idx = np.argmax(amplitudes)
        freq = freqs[max_amp_idx]
        am = amplitudes[max_amp_idx]
        sdv = phases[max_amp_idx]

        t1 = np.arange(0, N)
        rav0 = [model_a0 + am * np.sin((freq * t) - (sdv)) for t in t1]

        trend = [area[zn] - y[zn] for zn in range(len(area))]
        trend_format = [round(tr, 2) for tr in trend]

        optim = optimizer_f(y)
        raz = np.ptp(y)
        x1 = [0, raz / 4, 2 * math.pi / 12, 0]
        a0_opt, ampl_opt, freq_opt, phase_opt = optim.opt(x1, raz)
        optimal_params = optim.opt(x1, raz)
        fourier_equation_1 = f"y(t) = {a0_opt:.4f} + {ampl_opt:.4f} * sin({freq_opt:.4f} * t - ({phase_opt:.4f}))"

        znach_opt_furye = []
        sum_trend_i_optznach = []
        for t in t1:
            predicted_values = a0_opt + ampl_opt * np.sin(freq_opt * t - phase_opt)
            znach_opt_furye.append(round(predicted_values, 4))
            summa = znach_opt_furye[t] + trend_format[t]
            sum_trend_i_optznach.append(summa)

        y_true = areas[num_obl]
        y_pred = sum_trend_i_optznach

        r2_form_1 = r2_score(y_true, y_pred)
        r2 = round(r2_form_1, 3)

        return area, sum_trend_i_optznach, fourier_equation_1, r2,sum_trend_i_optznach,znach_opt_furye,a0_opt, ampl_opt, freq_opt, phase_opt

    def opt_sinus_2(self, amplitudes, freqs, phases, y, N, num_obl, area, areas):
        area, sum_trend_i_optznach, fourier_equation_1, r2_one, sum_trend_i_optznach,znach_opt_furye,a0_opt, ampl_opt, freq_opt, phase_opt = self.opt_sinus_1(amplitudes, freqs, phases, y, N,
                                                                                    num_obl, area, areas)
        t1 = np.arange(0, N)
        residuals = []
        for t in t1:
            residuals.append(y[t] - znach_opt_furye[t])

        optim_new = optimizer_f(residuals)
        raz_new = np.ptp(residuals)
        x1_new = [a0_opt, ampl_opt, freq_opt, phase_opt]

        a0_opt_n, ampl_opt_n, freq_opt_n, phase_opt_n = optim_new.opt(x1_new, raz_new)
        fourier_equation = f"y(t) = {a0_opt_n:.4f} + {ampl_opt_n:.4f} * sin({freq_opt_n:.4f} * t - ({phase_opt_n:.4f}))"

        znach_opt_furye_new = []
        sum_trend_i_optznach_new = []
        for t in t1:
            predicted_values_new = a0_opt_n + ampl_opt_n * np.sin(freq_opt_n * t - phase_opt_n)
            znach_opt_furye_new.append(round(predicted_values_new, 4))
            summa = znach_opt_furye_new[t] + sum_trend_i_optznach[t]
            sum_trend_i_optznach_new.append(summa)

        y_true = areas[num_obl]
        y_pred_new = sum_trend_i_optznach_new
        r2_form=r2_score(y_true, y_pred_new)
        r2_new = round(r2_form,3)

        return area, sum_trend_i_optznach_new, fourier_equation, r2_new

    def plot_optimization_results(self):
        areas = self.df.iloc[:, 1:].values.T
        print('это areas ',areas)
        self.selected_option = self.comboBox_for_opt.currentText()
        selected_column = self.comboBox_for_furye.currentText()
        t = np.arange(0,23)
        sum_opt=[]
        for column in self.df.columns:
            if column != 'Years':
                area = self.df[column].values
                zn = self.raschet_znach_trenda([area], t)  # вызов raschet_znach_trenda для получения zn
                amplitudes, freqs, phases, N, znach_razn, i, _ = self.furye(zn, [area], column, selected_column)

                if self.selected_option == "оптимизировать по 1 колебанию":
                    _, sum_trend_i_optznach, _, _ , _,_,_,_,_,_= self.opt_sinus_1(amplitudes, freqs, phases, znach_razn, N, i, area, [area])
                    sum_opt.append(sum_trend_i_optznach)
                    # self.plot_trend_optznachfurye(area, sum_trend_i_optznach, i, areas)
                elif self.selected_option == "оптимизировать по 2 колебаниям":
                    _, sum_trend_i_optznach_new, _, _ = self.opt_sinus_2(amplitudes, freqs, phases, znach_razn, N, i, area,
                                                                         [area])
                    sum_opt.append(sum_trend_i_optznach_new)
                    # self.plot_trend_optznachfurye_new(area, sum_trend_i_optznach_new, areas)

        self.plot_trend_optznachfurye(areas, sum_opt)
        self.setup_combobox_single_opt()



    def plot_trend_optznachfurye(self, areas, sum_trend_i_optznach_all):
        fig_new, axes = plt.subplots(nrows=1, ncols=len(areas), figsize=(15, 5))
        fig_new.suptitle("Оптимальная аппроксимация сигнала с помощью уравнения Фурье", fontsize=14, weight='bold',
                         family='Times New Roman')

        for i, (area, sum_trend_i_optznach) in enumerate(zip(areas, sum_trend_i_optznach_all)):
            axes[i].plot(area, color='green', label='Исходные')
            axes[i].plot(sum_trend_i_optznach, color='orange', label='Оптимальные значения с учетом 1/2 колебаний',
                         marker='o')
            axes[i].set_title(f'Область {i + 1}')
            # axes[i].legend()
            axes[i].set_xlabel('Год')
            axes[i].set_ylabel('Значения')

            for j, (x, y) in enumerate(zip(range(len(sum_trend_i_optznach)), sum_trend_i_optznach)):
                axes[i].annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center',
                                 fontweight='bold', size='6')

        self.canvas = FigureCanvas(fig_new)
        scene = QGraphicsScene()
        scene.addWidget(self.canvas)
        self.graph_for_opt.setScene(scene)

    def setup_combobox_single_opt(self):
        self.comboBox_single_opt.clear()
        column_names = self.df.columns[1:].values
        self.comboBox_single_opt.addItems(column_names)


    def open_single_graph_window(self, area, sum_trend_i_optznach,selected_area_name):
        self.single_graph_opt = SingleGraphWindow(area, sum_trend_i_optznach,selected_area_name)
        self.single_graph_opt.show()

    def plot_optimization_opt(self):

            selected_area_index = self.comboBox_single_opt.currentIndex()
            print('это индекс списка ',selected_area_index)
            selected_area_name = self.comboBox_single_opt.currentText()
            print("это что написано на списке ",selected_area_name)
            if selected_area_index >= 0:
                area = self.df[selected_area_name].values
                print('выбрано из списка значения: ',area)
                selected_option = self.comboBox_for_opt.currentText()
                selected_column = self.comboBox_for_furye.currentText()
                t = np.arange(0,23)
                zn = self.raschet_znach_trenda([area], t)  # вызов raschet_znach_trenda для получения zn
                amplitudes, freqs, phases, N, znach_razn, i, _ = self.furye(zn, [area], selected_area_name,
                                                                            selected_column)

                if (self.selected_option=='оптимизировать по 1 колебанию'):
                    _, sum_trend_i_optznach, _, _, _, _, _, _, _, _ = self.opt_sinus_1(amplitudes, freqs, phases,
                                                                                       znach_razn, N, i, area, [area])
                    self.open_single_graph_window(area, sum_trend_i_optznach,selected_area_name)
                if (self.selected_option=='оптимизировать по 2 колебаниям'):
                    _, sum_trend_i_optznach_new, _, _ = self.opt_sinus_2(amplitudes, freqs, phases,
                                                                                       znach_razn, N, i, area, [area])
                    self.open_single_graph_window(area, sum_trend_i_optznach_new,selected_area_name)




    def raschet_znach_trenda(self, areas, t):
        zn = []
        pred = []
        values_trend = []
        for i, area in enumerate(areas):
            pred1, coef1, intercept1,_,_ = self.linear_regression_analysis(t, area)
            pred.append(pred1)
            converted_data = [(coef1 * value + intercept1) for value in t]
            znach = [round(value, 2) for value in converted_data]
            print('\nЗначения, рассчитанные по уравнению тренда:\n ', znach)
            razn = [(area[j] - znach[j]) for j in range(len(areas[i]))]
            znach_razn = [round(value, 2) for value in razn]
            print('Разница между исходными данными и рассчитанными:\n', znach_razn)
            zn.append(znach_razn)
        print(zn)
        return zn
    def plot_selected_furye(self):
            selected_column = self.comboBox_for_furye.currentText()

            if selected_column:
                area = self.df[selected_column].values
                t = np.arange(0,23)
                nazvanie_col = self.df[selected_column].name
                areas = [area]
                zn = self.raschet_znach_trenda(areas, t)  # вызов raschet_znach_trenda для получения zn
                self.furye(zn, areas, area,nazvanie_col)

    def furye(self, zn, areas, area,nazvanie):

        for i, znach_razn in enumerate(zn):
            N = len(znach_razn)
            dt = 1 / N
            y_fft = rfft(znach_razn)
            freqs = rfftfreq(N, dt)
            amplitudes = np.abs(y_fft)
            phases = np.angle(y_fft)

            self.plot_furye(freqs, amplitudes, phases, znach_razn, y_fft, i,nazvanie)
            # self.opt_sinus(amplitudes, freqs, phases, znach_razn, N, i, areas[i], areas, axes, axes_2, fig, fig_1)

        return amplitudes, freqs, phases, N, znach_razn,i,areas

    def plot_furye(self, freqs, amplitudes, phases, znach_razn, y_fft, i, nazv):
        y_fft_1 = copy.deepcopy(y_fft)
        y_fft_2 = copy.deepcopy(y_fft)

        amplitudes = np.abs(y_fft_2)
        max_index = np.argmax(amplitudes)

        y_fft_max = np.zeros_like(y_fft_2)
        y_fft_max[max_index] = y_fft_2[max_index]

        new_sig = irfft(y_fft)
        new_sig2 = irfft(y_fft_max)

        self.fig3, axs = plt.subplots(3, 1, figsize=(8, 6))

        axs[0].stem(freqs, amplitudes, linefmt='blue', markerfmt='bo', basefmt=" ")
        axs[0].set_title(f'Амплитуды компонент ряда Фурье. {nazv} (область)')
        axs[0].set_xlabel('Гармоники')
        axs[0].set_ylabel('Амплитуда')

        axs[1].plot(new_sig2, color='green', label='Восстановлено')
        axs[1].plot(znach_razn, color='orange', label='Исходные данные', marker='o')
        axs[1].set_title(f'Частичное восстановление. {nazv} (область)')
        axs[1].set_xlabel('Год')
        axs[1].set_ylabel('Показатель')
        axs[1].legend()

        axs[2].plot(new_sig, color='red', label='Восстановлено')
        axs[2].plot(znach_razn, color='orange', label='Исходные данные', marker='o')
        axs[2].set_title(f'Полное восстановление. {nazv} (область)')
        axs[2].set_xlabel('Год')
        axs[2].set_ylabel('Показатель')
        axs[2].legend()

        self.fig3.subplots_adjust(hspace=0.6)
        self.update_graph_view(self.fig3)

    def update_graph_view(self, fig3):
        self.canvas = FigureCanvas(fig3)
        scene = QGraphicsScene()
        scene.addWidget(self.canvas)
        self.graph_furye.setScene(scene)
    def open_single_trend_window(self):
        selected_column = self.comboBox_for_trend.currentText()
        if selected_column:
            data = self.df[selected_column]
            years = np.arange(0,23)
            pred, _, _, _, _ = self.linear_regression_analysis(years, self.df[selected_column].values)
            self.single_graph_window = SingleGraphTrendWindow(self.df[selected_column].values, years, pred,data)
            self.single_graph_window.show()



    def linear_regression_analysis(self, years1, dan):
        model = LinearRegression()
        X = pd.DataFrame(years1)
        y = pd.DataFrame(dan)
        model.fit(X, y)
        pred1 = model.predict(X)
        coef = round(model.coef_[0][0], 4)
        intercept = round(model.intercept_[0], 4)
        koef_det = round(model.score(X, y), 4)
        kor_ot = round(math.sqrt(koef_det), 4)
        print("\nУравнение линейного тренда y(t) = {} + ({}) t".format(intercept, coef))
        print("Коэффициент детерминации (R²):", koef_det)
        print("Эмпирическое корреляционное отношение (√R²):", kor_ot)
        return pred1, coef, intercept, koef_det, kor_ot

    def plot_linear_regression_analysis(self, areas, t, pred):
        self.fig3, ax = plt.subplots(nrows=1, ncols=len(areas),figsize=(12,5))
        select_col=['Волгоградская облатсь','Ростовская область','Москва']
        for i, area in enumerate(areas):
            ax[i].plot(area, c='green', label='Исходные данные', marker='o')
            ax[i].plot(pred[i], label='Линия тренда', c='red')
            ax[i].set_xlabel('Год', fontsize=12)
            ax[i].set_ylabel('Показатель', fontsize=12)
            title = f'{select_col[i]}'
            ax[i].set_title(title, fontsize=10)
            self.fig3.suptitle("Линии тренда", fontsize=14)
            ax[i].legend(fontsize=8)

        self.canvas = FigureCanvas(self.fig3)
        # self.canvas.setFixedSize(1100, 500)
        scene = QGraphicsScene()
        scene.addWidget(self.canvas)
        self.graph_trend.setScene(scene)


    def graph_for_trend(self):
        print(self.df)
        self.comboBox_for_trend.addItems(self.df.columns[1:])
        years = np.arange(0,23)
        predictions = []
        for col in self.df.columns[1:]:
            pred, coef, intercept, koef_det, kor_ot = self.linear_regression_analysis(years, self.df[col])
            predictions.append(pred)
        self.plot_linear_regression_analysis(self.areas, years, predictions)

    def set_header_with_wrapping(self, table, headers):
        for i, header in enumerate(headers):
            label = QLabel(header)
            label.setWordWrap(True)
            label.setAlignment(Qt.AlignCenter)
            widget = QWidget()
            layout = QVBoxLayout()
            layout.addWidget(label)
            layout.setAlignment(Qt.AlignCenter)
            widget.setLayout(layout)
            table.setHorizontalHeaderItem(i, QTableWidgetItem())
            table.setHorizontalHeaderItem(i, QTableWidgetItem(header))

    def show_trend_results(self):
        years = np.arange(0,23)
        self.table_trend.setRowCount(0)
        self.table_trend.setColumnCount(4)
        headers = ["Область", "Уравнение тренда", "Коэффициент детерминации", "Эмпирическое корреляционное отношение"]

        self.set_header_with_wrapping(self.table_trend, headers)

        for i, col in enumerate(self.df.columns[1:]):
            pred, coef, intercept, koef_det, kor_ot = self.linear_regression_analysis(years, self.df[col])
            equation = f'y(t) = {intercept} + ({coef})t'
            self.table_trend.insertRow(i)
            self.table_trend.setItem(i, 0, QTableWidgetItem(f'Область {i + 1}'))
            self.table_trend.setItem(i, 1, QTableWidgetItem(equation))
            self.table_trend.setItem(i, 2, QTableWidgetItem(str(koef_det)))
            self.table_trend.setItem(i, 3, QTableWidgetItem(str(kor_ot)))

        self.table_trend.resizeColumnsToContents()
        # self.table_trend.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)


    def open_window_graph_rol(self):
        window = self.choosen_but()
        if window is not None:
            areas = self.areas
            combined_df = self.combined_df
            column_names = combined_df.columns[1:]
            dataf=self.df
            self.graph_rol_window = Roll_means(areas, combined_df, window,column_names,dataf)
            self.graph_rol_window.show()


    def choosen_but(self):
        if self.rbut_3.isChecked():
            window_size = 3
        elif self.rbut_5.isChecked():
            window_size = 5
        return window_size
    def load_rolling_means(self):
        if self.rbut_3 is not None and self.rbut_5 is not None:
            window = self.choosen_but()
            self.combined_df,self.all_df = rolling_means(self.headers,self.areas, window)
        self.display_df_in_table(self.combined_df)

    def display_df_in_table(self, df):
        self.table_roll.setColumnCount(len(df.columns))
        self.table_roll.setHorizontalHeaderLabels(df.columns)
        header = self.table_roll.horizontalHeader()
        self.table_roll.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        # for i in range(len(df.columns)):
            # header.setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)
            # self.table_roll.setColumnWidth(i, 230)

        self.table_roll.setRowCount(len(df))
        for i in range(len(df)):
            for j in range(len(df.columns)):
                self.table_roll.setItem(i, j, QTableWidgetItem(str(df.iat[i, j])))


    def keep_as_table(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Table Data", "", "CSV Files (*.csv);;All Files (*)",
                                                   options=options)
        if file_name:
            self.save_table_data(file_name)
            self.show_message_table("Файл сохранен!")

    def keep_as_table_roll(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Table Data", "", "CSV Files (*.csv);;All Files (*)",
                                                   options=options)
        if file_name:
            self.save_table_roll(file_name)
            self.show_message_table("Файл сохранен!")
    def save_table_data(self, file_name):
        row_count = self.table_for_indinam.rowCount()
        column_count = self.table_for_indinam.columnCount()
        with open(file_name, 'w', newline='',encoding='utf-8') as file:
            writer = csv.writer(file)
            headers = [self.table_for_indinam.horizontalHeaderItem(i).text() for i in range(column_count)]
            writer.writerow(headers)

            for row in range(row_count):
                row_data = []
                for column in range(column_count):
                    item = self.table_for_indinam.item(row, column)
                    row_data.append(item.text() if item else '')
                writer.writerow(row_data)

    def save_table_roll(self, file_name):
        row_count = self.table_roll.rowCount()
        column_count = self.table_roll.columnCount()
        with open(file_name, 'w', newline='',encoding='utf-8') as file:
            writer = csv.writer(file)
            headers = [self.table_roll.horizontalHeaderItem(i).text() for i in range(column_count)]
            writer.writerow(headers)

            for row in range(row_count):
                row_data = []
                for column in range(column_count):
                    item = self.table_roll.item(row, column)
                    row_data.append(item.text() if item else '')
                writer.writerow(row_data)

    def show_message_table(self, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText(message)
        msg_box.setWindowTitle("Сообщение")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()
    def res_tabl_ind_dinam(self):
        area_key = f'Область {self.comboBox.currentIndex()}'
        if self.but_rost.isChecked():
            key = "Темпы роста"
        elif self.but_prirost.isChecked():
            key = "Темпы прироста"
        elif self.but_absrost.isChecked():
            key = "Абсолютные приросты"
        else:
            return

        daf = self.dataframes[area_key][key]
        self.update_table(daf)

    def open_graph_window(self):
        indicator = self.comboBox.currentText()
        area_key = f'Область {self.comboBox.currentIndex()}'
        if self.but_rost.isChecked():
            key = "Темпы роста"
        elif self.but_prirost.isChecked():
            key = "Темпы прироста"
        elif self.but_absrost.isChecked():
            key = "Абсолютные приросты"
        else:
            return

        df = self.dataframes[area_key][key]
        self.plot_graph_other(df, f'{indicator} - {key}','Год',f'{key}')
    def plot_graph_other(self, df, title, xlabel,ylabel):
        data1 = df.iloc[:, 1].values
        data2 = df.iloc[:, 2].values

        self.ind_window = IndDinam(data1, data2, title,xlabel,ylabel)
        self.ind_window.show()
    def update_table(self,df):

        self.table_for_indinam.setColumnCount(len(df.columns))
        self.table_for_indinam.setHorizontalHeaderLabels(df.columns)
        header = self.table_for_indinam.horizontalHeader()
        self.table_for_indinam.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

        # self.table_for_indinam.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        # for i in range(len(df.columns)):
        #     header.setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)

        self.table_for_indinam.setRowCount(len(df))
        for i in range(len(df)):
            for j in range(len(df.columns)):
                self.table_for_indinam.setItem(i, j, QTableWidgetItem(str(df.iat[i, j])))

    def show_analysis_tab(self):
        self.tabWidget.setCurrentIndex(1)

    def show_instruction(self):
        self.label.show()

    def hide_instruction(self):
        self.label.deleteLater()


    def open_csv_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)",
                                                       options=options)
        if file_name:
            self.load_csv_data(file_name)
        return file_name

    def load_csv_data(self, file_name):
        try:
            dataframe = pd.read_csv(file_name, delimiter=',')
            if dataframe.empty:
                raise ValueError("CSV файл пуст.")
            if not isinstance(dataframe, pd.DataFrame):
                raise ValueError("Не удалось загрузить CSV файл в DataFrame.")

            self.df = dataframe
            self.tableWidget.setColumnCount(len(self.df.columns))
            self.tableWidget.setHorizontalHeaderLabels(self.df.columns)
            self.tableWidget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
            self.tableWidget.setRowCount(len(self.df))
            for i in range(len(self.df)):
                for j in range(len(self.df.columns)):
                    self.tableWidget.setItem(i, j, QTableWidgetItem(str(self.df.iat[i, j])))

            self.hide_instruction()
            ch_row = len(self.df)
            ch_col = len(self.df.columns)
            name_file = os.path.basename(file_name)
            self.update_text_browser(ch_row, ch_col, name_file)

            self.comboBox.clear()
            self.comboBox.addItem("")
            if len(self.df.columns) > 1:
                self.comboBox.addItems(self.df.columns[1:])
            self.comboBox.setCurrentIndex(0)

            self.areas, self.years, self.headers = nach_dan(file_name)
            self.dataframes = create_dataframes(self.areas, self.years)

        except Exception as e:
            QMessageBox.critical(self, "Ошибка загрузки файла", f"Не удалось загрузить файл: {str(e)}")


    def update_text_browser(self, num_rows, num_columns,name_file):
            self.for_ch_rows_2.setText(f" {num_rows}")
            self.for_ch_col_2.setText(f" {num_columns}")
            self.for_name_file.setText(f" {name_file}")
            self.for_name_file.setWordWrap(True)

    def handle_combobox_change(self, index):
        self.comboBox.setEditable(True)
        self.comboBox.lineEdit().setReadOnly(True)
        self.comboBox.lineEdit().setAlignment(Qt.AlignCenter)
        self.comboBox.lineEdit().setPlaceholderText("Выберите признак")
        font = QFont("Times New Roman", 12, QFont.Bold)
        self.comboBox.setFont(font)
        self.comboBox.lineEdit().setFont(font)
        if index == 0:
            self.comboBox.lineEdit().setPlaceholderText("Выберите признак")
        else:
            self.comboBox.lineEdit().setPlaceholderText("")

class SingleGraphWindow(QtWidgets.QDialog):
    def __init__(self, area, sum_trend_i_optznach, selected_area_name,parent=None):
        super(SingleGraphWindow, self).__init__(parent)
        uic.loadUi('D:\\STUDY\\симонов\\для диплома\\для приложения\\single_graph_opt.ui', self)
        self.graph_for_single_opt = self.findChild(QtWidgets.QGraphicsView, 'graph_for_single_opt')
        self.keep_as=self.findChild(QtWidgets.QPushButton,'keep_as')
        self.buton_closw=self.findChild(QtWidgets.QPushButton,'buton_closw')
        self.keep_as.clicked.connect(self.keep_graph)
        self.buton_closw.clicked.connect(self.close)
        print('area какие получает новый класс ',area)
        print('sum_opt какие получает новый класс ', sum_trend_i_optznach)
        if not self.graph_for_single_opt:
            print("Graphics view not found")
            return
        self.plot_trend_optznachfurye_opt(area, sum_trend_i_optznach,selected_area_name)


    def keep_graph(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Graph", "", "PNG Files (*.png);;All Files (*)",
                                                   options=options)
        if file_name:
            self.save_graph_data(file_name)
            self.show_message("График сохранен!")

    def save_graph_data(self, file_name):
        self.fig.savefig(file_name)

    def show_message(self, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText(message)
        msg_box.setWindowTitle("Сообщение")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()
    def plot_trend_optznachfurye_opt(self, area, sum_trend_i_optznach,selected_area_name):
        self.fig, ax = plt.subplots()
        plt.rc('font', family='Times New Roman')

        self.canvas = FigureCanvas(self.fig)
        scene = QGraphicsScene()
        scene.addWidget(self.canvas)
        self.graph_for_single_opt.setScene(scene)

        ax.plot(area, color='green', label='Исходные')
        ax.plot(sum_trend_i_optznach, color='orange', label='Оптимальные значения', marker='o')
        ax.set_title(f'Оптимальная аппроксимация {selected_area_name}', fontsize=14, weight='bold', family='Times New Roman')
        ax.set_xlabel('Год')
        ax.set_ylabel('Значения')
        ax.legend()
        for j, (x, y) in enumerate(zip(range(len(sum_trend_i_optznach)), sum_trend_i_optznach)):
            ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold', size='6')




class IndDinam(QtWidgets.QDialog):
    def __init__(self, data1, data2, title,xlabel,ylabel, parent=None):
        super(IndDinam, self).__init__(parent)
        uic.loadUi("D:\\STUDY\\симонов\\для диплома\\для приложения\\graph_indinam.ui", self)
        # self.button_analysis.clicked.connect(self.text1)
        self.graph_indinam=self.findChild(QtWidgets.QGraphicsView,'graph_indinam')
        self.plot_graph(data1, data2, title,xlabel,ylabel)
        self.but_gra_clos=self.findChild(QtWidgets.QPushButton,'but_gra_clos')
        self.but_gra_clos.clicked.connect(self.close)
        self.but_gra = self.findChild(QtWidgets.QPushButton, 'but_gra')
        self.but_gra.clicked.connect(self.keep_graph)

    def keep_graph(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Graph", "", "PNG Files (*.png);;All Files (*)",
                                                   options=options)
        if file_name:
            self.save_graph_data(file_name)
            self.show_message("График сохранен!")
    def save_graph_data(self, file_name):
        self.fig.savefig(file_name)
    def show_message(self, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText(message)
        msg_box.setWindowTitle("Сообщение")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()


    def plot_graph(self, data1, data2, title,xlabel,ylabel):
        self.graph_indic_dinamic(data1, data2, title,xlabel,
                            ylabel, color1='red', color2='blue', label1='Цепной темп',
                            label2='Базисный темп')

    def graph_indic_dinamic(self,data1, data2, title, xlabel='Год', ylabel='Значение', color1='red', color2='blue',
                                label1='Цепной', label2='Базисный'):
        plt.rc('font', family='Times New Roman')
        self.fig, ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setFixedSize(670,378)
        scene = QGraphicsScene()
        scene.addWidget(self.canvas)
        self.graph_indinam.setScene(scene)
        years = np.arange(len(data2))
        ax.plot(data1, color=color1, label=label1, marker='o')
        ax.plot(data2, color=color2, label=label2, marker='s')
        ax.set_xticks(years)
        ax.set_xticklabels(years, rotation=90)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10)
        ax.axhline(0, linewidth=0.5, color='black')
        ax.axvline(0, linewidth=0.5, color='black')

        for i, (x, y) in enumerate(zip(range(len(data1)), data1)):
            ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center',
                            fontweight='bold', size='9')
        for i, (x, y) in enumerate(zip(range(len(data2)), data2)):
            ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, -15), ha='center',
                            fontweight='bold', size='9')
        plt.style.use('seaborn-v0_8')


class Roll_means(QtWidgets.QDialog):
    def __init__(self, areas, combined_df, window_size,column_names, df, parent=None):
        super(Roll_means, self).__init__(parent)
        uic.loadUi("D:\\STUDY\\симонов\\для диплома\\для приложения\\graph_rollmeans.ui", self)
        # self.button_analysis.clicked.connect(self.text1)
        self.graph_roll_all=self.findChild(QtWidgets.QGraphicsView,'graph_roll_all')
        self.plot_graph_roll(areas, combined_df, column_names)
        self.but_close=self.findChild(QtWidgets.QPushButton,'but_close')
        self.but_close.clicked.connect(self.close)
        self.but_keep_as = self.findChild(QtWidgets.QPushButton, 'but_keep_as')
        self.but_keep_as.clicked.connect(self.keep_graph)
        self.df=combined_df
        self.areas = areas
        self.col_n = column_names
        # combobox
        self.comboBox_for_rol = self.findChild(QtWidgets.QComboBox, 'comboBox_for_rol')
        self.comboBox_for_rol.setEditable(True)
        self.comboBox_for_rol.lineEdit().setReadOnly(True)
        self.comboBox_for_rol.lineEdit().setAlignment(Qt.AlignCenter)
        self.comboBox_for_rol.lineEdit().setPlaceholderText("Выберите признак")
        self.comboBox_for_rol.currentIndexChanged.connect(self.handle_combobox_change)
        self.update_combobox_for_rol(column_names)
        self.show_single_graph = self.findChild(QtWidgets.QPushButton, 'show_single_graph')
        self.show_single_graph.clicked.connect(self.open_window_single_graph_rol)


    def open_window_single_graph_rol(self):
        index = self.comboBox_for_rol.currentIndex()
        if index != -1:
            dframe = self.df.iloc[:, index + 1]
            area = self.areas[index]
            self.singl_graph_rol_window = Single_roll_means(dframe, area)
            self.singl_graph_rol_window.show()


    def update_combobox_for_rol(self, column_names):
        self.comboBox_for_rol.clear()
        self.comboBox_for_rol.addItems(column_names)

    def handle_combobox_change(self, index):
        if index == 0:
            self.comboBox_for_rol.lineEdit().setPlaceholderText("Выберите признак")
        else:
            self.comboBox_for_rol.lineEdit().setPlaceholderText("")
        self.comboBox_for_rol.setEditable(True)
        self.comboBox_for_rol.lineEdit().setReadOnly(True)
        self.comboBox_for_rol.lineEdit().setAlignment(Qt.AlignCenter)
        font = QFont("Times New Roman", 12, QFont.Bold)
        self.comboBox_for_rol.setFont(font)
        self.comboBox_for_rol.lineEdit().setFont(font)
    def keep_graph(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Graph", "", "PNG Files (*.png);;All Files (*)",
                                                   options=options)
        if file_name:
            self.save_graph_data(file_name)
            self.show_message("График сохранен!")

    def save_graph_data(self, file_name):
        self.fig.savefig(file_name)

    def show_message(self, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText(message)
        msg_box.setWindowTitle("Сообщение")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

    def plot_graph_roll(self, areas, combined_df, column_names):
        self.plot_rolling_means_combined(areas,combined_df,column_names)

    def plot_rolling_means_combined(self,areas, combined_df, column_names):
            print(combined_df)
            plt.rc('font', family='Times New Roman')
            self.fig, ax = plt.subplots(nrows=1, ncols=len(areas), figsize=(14, 5))
            self.canvas = FigureCanvas(self.fig)
            # Устанавливаем размеры canvas по размеру QGraphicsView
            # self.canvas.setFixedSize(1100, 500)
            scene = QGraphicsScene()
            scene.addWidget(self.canvas)
            self.graph_roll_all.setScene(scene)
            for i, area in enumerate(areas):
                years = combined_df['Год']
                rolling_mean = combined_df[f'{column_names[i]}']
                ax[i].plot(area, c='black', label='Исходные данные')
                ax[i].plot(rolling_mean, c='red', label=f'Скользящее среднее', marker='o')
                ax[i].set_xlabel('Год', fontsize=12)
                ax[i].set_ylabel('Скользящие средние', fontsize=12)
                title = f'{column_names[i]}'
                ax[i].set_title(title, fontsize=14)
                ax[i].legend(fontsize=10)
                ax[i].set_xticks(years)
                ax[i].set_xticklabels(years, rotation=90)
            self.fig.suptitle("Скользящие средние", fontsize=14, weight='bold', family='Times New Roman')

class Single_roll_means(QtWidgets.QDialog):
        def __init__(self, data_column, area,parent=None):
            super(Single_roll_means, self).__init__(parent)
            uic.loadUi("D:\\STUDY\\симонов\\для диплома\\для приложения\\single_graph_rolmeans.ui", self)
            # self.button_analysis.clicked.connect(self.text1)
            self.graphicsView = self.findChild(QtWidgets.QGraphicsView, 'graphicsView')
            self.plot_rolling_means_single(data_column,area)
            self.button_close = self.findChild(QtWidgets.QPushButton, 'button_close')
            self.button_close.clicked.connect(self.close)
            self.keep_as = self.findChild(QtWidgets.QPushButton, 'keep_as')
            self.keep_as.clicked.connect(self.keep_graph)


        def plot_rolling_means_single(self, data_column, area):
            plt.rc('font', family='Times New Roman')
            fig, ax = plt.subplots(figsize=(8, 5))
            self.fig = fig
            self.canvas = FigureCanvas(self.fig)
            scene = QGraphicsScene()
            scene.addWidget(self.canvas)
            self.graphicsView.setScene(scene)

            years = data_column.index
            roll_mean = data_column

            ax.plot(area, c='yellow', label='Исходные данные')
            ax.plot(roll_mean, c='red', label='Скользящее среднее', marker='o')
            ax.set_xlabel('Год', fontsize=10)
            ax.set_ylabel('Скользящие средние', fontsize=10)
            title = f'{data_column.name} (область)'
            ax.set_title(title, fontsize=12)
            ax.legend(fontsize=8)
            ax.set_xticks(years)
            ax.set_xticklabels(years, rotation=90)
            fig.suptitle("Скользящие средние", fontsize=14, weight='bold', family='Times New Roman')


        def keep_graph(self):
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Graph", "", "PNG Files (*.png);;All Files (*)",
                                                       options=options)
            if file_name:
                self.save_graph_data(file_name)
                self.show_message("График сохранен!")

        def save_graph_data(self, file_name):
            self.fig.savefig(file_name)

        def show_message(self, message):
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setText(message)
            msg_box.setWindowTitle("Сообщение")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec_()

class SingleGraphTrendWindow(QtWidgets.QDialog):
    def __init__(self, area, t, pred, data,parent=None):
        super(SingleGraphTrendWindow, self).__init__(parent)
        uic.loadUi("D:\\STUDY\\симонов\\для диплома\\для приложения\\single_graph_trend.ui", self)
        self.graphicsView=self.findChild(QtWidgets.QGraphicsView,'graphicsView')
        self.sing_trend_save=self.findChild(QtWidgets.QPushButton,'sing_trend_save')
        self.sing_trend_save.clicked.connect(self.keep_graph)
        self.but_sing_close_trend=self.findChild(QtWidgets.QPushButton,'but_sing_close_trend')
        self.but_sing_close_trend.clicked.connect(self.close)
        self.plot_single_graph(area, t, pred,data)

    def keep_graph(self):
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Graph", "", "PNG Files (*.png);;All Files (*)",
                                                       options=options)
            if file_name:
                self.save_graph_data(file_name)
                self.show_message("График сохранен!")

    def save_graph_data(self, file_name):
            self.fig_tr.savefig(file_name)

    def show_message(self, message):
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setText(message)
            msg_box.setWindowTitle("Сообщение")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec_()
    def plot_single_graph(self, area, t, pred,data):
            self.fig_tr, ax = plt.subplots()
            self.canvas = FigureCanvas(self.fig_tr)
            # self.canvas.setFixedSize(720, 365)
            scene = QGraphicsScene()
            scene.addWidget(self.canvas)
            self.graphicsView.setScene(scene)

            ax.plot(t, area, c='green', label='Исходные данные', marker='o')
            ax.plot(t, pred, label='Линия тренда', c='red')
            ax.set_xlabel('Год', fontsize=12)
            ax.set_ylabel('Показатель', fontsize=12)
            title = f'График тренда. {data.name}'
            ax.set_title(title, fontsize=14)
            ax.legend(fontsize=8)

def main():
    app = QtWidgets.QApplication(sys.argv)  # Новый экземпляр QApplication
    window = MainWindow()  #  объект класса ExampleApp
    window.show()
    app.exec_()


if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()