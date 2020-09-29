#!/usr/bin/env python3
import numpy as np

class FileParser:
        def _outputs_normalizer(outputs, max_value, min_value):
            for idx in  range(len(outputs)):
                outputs[idx] = (outputs[idx] - min_value) / (max_value - min_value)
            return [outputs, min_value, max_value]

        @staticmethod
        def data_parser():
            entries_file = open("/Users/nachograsso/Desktop/ITBA/SIA/SIA-TP3/SIA-TP3/conjunto-entrenamiento-ej2.txt", "r")
            [output, min_value, max_value] = FileParser.__outputs_parser()
            data = []
            i = 0
            for l_i in entries_file:
                entry = []
                entry.append(1.0)
                entry += list(map(float, l_i.split()))
                entry.append(output[i])
                data.append(entry)
                i += 1
            return FileParser.__get_both_datas(data) + [min_value, max_value]

        def __get_both_datas(data):
            size = 40
            test_data = []
            for i in range(size):
                idx = np.random.randint(0, len(data))
                entry = data[idx]
                test_data.append(entry)
                data.remove(entry)
            return [data, test_data]

        def __outputs_parser():
            output_file = open("/Users/nachograsso/Desktop/ITBA/SIA/SIA-TP3/SIA-TP3/salida-esperada-ej2.txt", "r")
            outputs = []
            max_value = 0
            min_value = 10000
            for l_i in output_file:
                output = float(l_i.split()[0])
                outputs.append(output)
                max_value = max(max_value, output)
                min_value = min(min_value, output)

            return [outputs, min_value, max_value]


        @staticmethod
        def mlp_entries_parser(rows, cols):
            entries_file = open('/Users/nachograsso/Desktop/ITBA/SIA/SIA-TP3/SIA-TP3/TP3-ej3-mapa-de-pixeles-digitos-decimales.txt')
            entries = []
            count = 0
            e = ''
            for l_i in entries_file:
                entry = l_i.split()
                if (count + len(entry)) <= rows * cols:
                    for i in range(len(entry)):
                        e += (entry[i] + ' ')
                        count +=1
                if count >= 35:
                    entries.append(e)
                    count = 0
                    e = ''
            return entries
