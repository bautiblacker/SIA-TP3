import numpy as np

class FileParser:
        def _outputs_normalizer(outputs, max_value, min_value):
            for idx in  range(len(outputs)):
                outputs[idx] = (outputs[idx] - min_value) / (max_value - min_value)

            return outputs

        @staticmethod
        def entries_parser():
            entries_file = open("conjunto-entrenamiento-ej2.txt", "r")
            entries = []
            for l_i in entries_file:
                entry = list(map(float, l_i.split()))
                # #  ver esto (!!) #
                entry.append(1.0)
                # # # # # # # # # #

            return entries


        @staticmethod
        def outputs_parser():
            output_file = open("salida-esperada-ej2.txt", "r")
            outputs = []
            max_value = 0
            min_value = 10000
            for l_i in output_file:
                output = float(l_i.split()[0])
                outputs.append(output)
                max_value = max(max_value, output)
                min_value = min(min_value, output)

            return FileParser._outputs_normalizer(outputs, max_value, min_value)


