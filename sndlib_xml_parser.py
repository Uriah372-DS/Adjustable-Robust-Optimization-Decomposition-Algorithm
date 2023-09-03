from collections import OrderedDict

import xmltodict
import networkx as nx


class SndlibXmlParser:
    str_keys = [
        'xmlns', 'version', 'granularity', 'unit', 'origin', 'coordinatesType', 'id', 'source', 'target',  # network
        'state', 'demandId',  # solution
        'nodeModel', 'linkModel', 'linkCapacityModel', 'fixedChargeModel',
        'demandModel', 'routingModel', 'admissiblePathModel', 'hopLimitModel',
        'objectiveModel', 'survivabilityModel'  # model
    ]
    numeric_keys = [
        'time', 'x', 'y', 'setupCost', 'routingCost', 'capacity', 'cost', 'demandValue', 'maxPathLength',  # network
        'installCount', 'flowPathValue'  # solution
    ]
    list_keys = [
        'node', 'link', 'addModule', 'demand',  # network
        'linkConfiguration', 'demandRouting', 'flowPath', 'routingPath'  # solution
    ]
    path_dependent = [
        'linkId'
    ]

    last_parsed_object_type = None

    # noinspection PyUnusedLocal
    @staticmethod
    def xmltodict_postprocessor(path, key, value):

        key = str(key).replace('@', '')

        if str(key) in SndlibXmlParser.str_keys:
            return key, str(value)

        elif str(key) in SndlibXmlParser.numeric_keys:
            try:
                value = float(value)
                if str(value).endswith('.0'):
                    return key, int(str(value).replace('.0', ''))
                else:
                    return key, value
            except ValueError:
                return key, str(value)

        else:
            return key, value

    @staticmethod
    def lists_postprocessor(raw_parsed_dict: dict) -> dict:
        if SndlibXmlParser.last_parsed_object_type == 'network':
            # 'addModule':
            for i in range(len(raw_parsed_dict['network']['networkStructure']['links']['link'])):
                link_module = raw_parsed_dict['network']['networkStructure']['links']['link'][i]
                if 'additionalModules' in link_module.keys() \
                        and not isinstance(link_module['additionalModules']['addModule'], list):
                    link_module['additionalModules']['addModule'] = [link_module['additionalModules']['addModule'], ]

        elif SndlibXmlParser.last_parsed_object_type == 'solution':
            # 'flowPath':
            for i in range(len(raw_parsed_dict['solution']['demandRoutings']['demandRouting'])):
                if not isinstance(raw_parsed_dict['solution']['demandRoutings']['demandRouting'][i]['flowPath'], list):
                    raw_parsed_dict['solution']['demandRoutings']['demandRouting'][i]['flowPath'] = \
                        [raw_parsed_dict['solution']['demandRoutings']['demandRouting'][i]['flowPath'], ]
            # 'linkId':
            print(raw_parsed_dict['solution']['demandRoutings']['demandRouting'][0]['flowPath'])
            for i in range(len(raw_parsed_dict['solution']['demandRoutings']['demandRouting'])):
                for j in range(len(raw_parsed_dict['solution']['demandRoutings']['demandRouting'][i]['flowPath'])):
                    if not isinstance(raw_parsed_dict['solution']['demandRoutings']['demandRouting'][i]['flowPath'][j][
                                          'routingPath']['linkId'], list):
                        raw_parsed_dict['solution']['demandRoutings']['demandRouting'][i]['flowPath'][j]['routingPath'][
                            'linkId'] = \
                            [raw_parsed_dict['solution']['demandRoutings']['demandRouting'][i]['flowPath'][j][
                                 'routingPath']['linkId'], ]

        return raw_parsed_dict

    @staticmethod
    def xml_to_dict(xml_path: str):
        with open(xml_path, 'r') as f:
            parsed_dict = xmltodict.parse(
                xml_input=f.read(),
                postprocessor=SndlibXmlParser.xmltodict_postprocessor
            )
            object_type = list(parsed_dict.keys())[0]
            if object_type not in ['network', 'solution', 'model']:
                raise ValueError("Not a supported object in SNDlib")
            SndlibXmlParser.last_parsed_object_type = object_type
            parsed_dict = SndlibXmlParser.lists_postprocessor(raw_parsed_dict=parsed_dict)

            return parsed_dict[SndlibXmlParser.last_parsed_object_type]

    @staticmethod
    def dict_to_network(data_dict: dict):
        network = nx.DiGraph()
        if data_dict != {}:

            # global properties:
            data_dict['coordinatesType'] = data_dict['networkStructure']['nodes']['coordinatesType']
            del data_dict['networkStructure']['nodes']['coordinatesType']
            for key in ['xmlns', 'version', 'meta', 'coordinatesType']:
                if key in data_dict.keys():
                    network.graph.update({key: data_dict[key]})

            # nodes:
            for node in data_dict['networkStructure']['nodes']['node']:
                node_attributes = {'x': float(node['coordinates']['x']), 'y': float(node['coordinates']['y'])}
                network.add_node(node_for_adding=node['id'], **node_attributes)

            # links (edges):
            pre_installed_link_capacities = 0
            pre_installed_link_costs = 0
            for link in data_dict['networkStructure']['links']['link']:

                link_attributes = {'id': link['id']}

                # costs:
                if 'setupCost' in link.keys():
                    link_attributes['setupCost'] = link['setupCost']

                if 'routingCost' in link.keys():
                    link_attributes['routingCost'] = link['routingCost']

                # pre-installed cost & capacity module:
                if 'preInstalledModule' in link.keys():
                    pre_installed_link_capacities += link['preInstalledModule']['capacity']
                    pre_installed_link_costs += link['preInstalledModule']['cost']

                # additional cost & capacity modules:
                link_modules = []
                if 'additionalModules' in link.keys():
                    for m in link['additionalModules']['addModule']:
                        link_modules.append(m)
                link_attributes['capacity'] = sum([link_modules[m]['capacity'] for m in range(len(link_modules))])
                link_attributes['cost'] = sum([link_modules[m]['cost'] for m in range(len(link_modules))])

                network.add_edge(u_of_edge=link['source'],
                                 v_of_edge=link['target'],
                                 **link_attributes)
            network.graph['totalPreInstalledCapacities'] = pre_installed_link_capacities
            network.graph['totalPreInstalledCosts'] = pre_installed_link_costs

            # demands (commodities):
            commodities = OrderedDict()
            for commodity in data_dict['demands']['demand']:
                commodities[commodity['id']] = OrderedDict()
                commodities[commodity['id']]['source'] = commodity['source']
                commodities[commodity['id']]['target'] = commodity['target']
                commodities[commodity['id']]['demand'] = commodity['demandValue']

                # optionals:
                if 'maxPathLength' in commodity.keys():
                    commodities[commodity['id']]['maxPathLength'] = commodity['maxPathLength']
                if 'admissiblePaths' in commodity.keys():
                    commodities[commodity['id']]['admissiblePaths'] = commodity['admissiblePaths']['admissiblePath']

            network.graph['Commodities'] = commodities

        return network

    @staticmethod
    def parse_network(xml_path: str = "",
                      k_top_demands=None,
                      biggest: bool = True) -> nx.DiGraph:
        if xml_path != "":
            network_dict = SndlibXmlParser.xml_to_dict(xml_path)

            if k_top_demands is not None:
                if biggest:
                    keep = sorted(network_dict['demands']['demand'],
                                  key=lambda x: x['demandValue'])[-k_top_demands:]
                else:
                    keep = sorted(network_dict['demands']['demand'],
                                  key=lambda x: x['demandValue'])[:k_top_demands]
                remove_commodities = []
                for d in network_dict['demands']['demand']:
                    if d not in keep:
                        remove_commodities.append(d)
                for commodity in remove_commodities:
                    network_dict['demands']['demand'].remove(commodity)

            return SndlibXmlParser.dict_to_network(network_dict)
