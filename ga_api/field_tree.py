class FieldNode():
    def __setitem__(self, path, args):
        idx = args[0]
        dsc = args[1]
        setattr(self, '_name', '@'+'.'.join(path[:idx]))
        if idx == len(path):
            setattr(self, '_desc', dsc)
        elif hasattr(self, path[idx]):
            getattr(self, path[idx])[path] = (idx+1, dsc)
        else:
            setattr(self, path[idx], FieldNode())
            getattr(self, path[idx])[path] = (idx+1, dsc)

    def __call__(self):
        return self._name

    def __repr__(self):
        fields = []
        for a in dir(self):
            attr = getattr(self, a)
            if isinstance(attr, FieldNode):
                fields.append(str(attr))
        if len(fields) == 0:
            return f'{self._name} => {self._desc}'
        else:
            return '\n'.join(fields)

class FieldTable():
    def __init__(self):
        self.node = FieldNode()

    def load(self, conf):
        for name, desc in conf.items():
            path = name.split('.')
            args = (0, desc)
            self.node[path] = args
        return self.node

DefaultConf = {
    "meta.uid"                     : "Record UID"                                               ,
    "meta.time"                    : "Record Timestamp"                                         ,
    "flow.addr.src"                : "Source IP"                                                ,
    "flow.addr.dst"                : "Destination IP"                                           ,
    "flow.protocol"                : "IP Protocol Number"                                       ,
    "flow.port.src"                : "Source TCP/UDP Port"                                      ,
    "flow.port.dst"                : "Destination TCP/UDP Port"                                 ,
    "flow.tcp_flags"               : "TCP Flags"                                                ,
    "flow.flows"                   : "Scaled Flow Count"                                        ,
    "flow.packets"                 : "Scaled Packet Count"                                      ,
    "flow.bytes"                   : "Scaled Byte Count"                                        ,
    "flow.exp_addr"                : "Exporter Router IP"                                       ,
    "flow.nexthop"                 : "Next Hop Router IP"                                       ,
    "flow.interface.in"            : "ifIndex of Input Interface"                               ,
    "flow.interface.out"           : "ifIndex of Output Interface"                              ,
    "flow.prefix_len.src"          : "Prefix Length of Source IP"                               ,
    "flow.prefix_len.dst"          : "Prefix Length of Destination IP"                          ,
    "flow.as.src"                  : "AS Number of Source IP"                                   ,
    "flow.as.dst"                  : "AS Number of Destination IP"                              ,
    "flow.sample_rate"             : "Sampling Rate"                                            ,
    "flow.first"                   : "Time of First Packet"                                     ,
    "flow.last"                    : "Time of Last Packet"                                      ,
    "flow.resample_rate"           : "Resampling Rate"                                          ,
    "flow.cos.in"                  : "Input IP CoS"                                             ,
    "flow.cos.out"                 : "Output IP CoS"                                            ,
    "bgp.as_path.src"              : "BGP AS Path of Source IP"                                 ,
    "bgp.as_path.dst"              : "BGP AS Path of Destination IP"                            ,
    "bgp.community.src"            : "BGP Community of Source IP"                               ,
    "bgp.community.dst"            : "BGP Community of Destination IP"                          ,
    "bgp.nexthop.src"              : "BGP Next Hop IP of Source IP"                             ,
    "bgp.nexthop.dst"              : "BGP Next Hop IP of Destination IP"                        ,
    "bgp.neighbor_as.src"          : "BGP Neighbor ASN of Source IP"                            ,
    "bgp.neighbor_as.dst"          : "BGP Neighbor ASN of Destination IP"                       ,
    "controller.router"            : "Tagged Router Instance"                                   ,
    "controller.interface.in"      : "Tagged Interface Input Instance"                          ,
    "controller.interface.out"     : "Tagged Interface Output Instance"                         ,
    "controller.home.src"          : "Tagged Home Source Instances"                             ,
    "controller.home.dst"          : "Tagged Home Destination Instances"                        ,
    "controller.neighbor.src"      : "Tagged Neighbor Source Instances"                         ,
    "controller.neighbor.dst"      : "Tagged Neighbor Destination Instances"                    ,
    "controller.subnetwork.src"    : "Tagged Sub-Network Source Instances"                      ,
    "controller.subnetwork.dst"    : "Tagged Sub-Network Destination Instances"                 ,
    "controller.serverfarm.src"    : "Tagged Server Farm Source Instances"                      ,
    "controller.serverfarm.dst"    : "Tagged Server Farm Destination Instances"                 ,
    "controller.filter.forward"    : "Tagged Filter Forward Instances"                          ,
    "controller.filter.opposite"   : "Tagged Filter Opposite Instances"                         ,
    "controller.boundary.forward"  : "Tagged Boundary Forward Instances"                        ,
    "controller.boundary.opposite" : "Tagged Boundary Opposite Instances"                       ,
    "nat.nat_type.src"             : "NAT Type of Source IP (0:none, 1:inside, 2:outside)"      ,
    "nat.nat_type.dst"             : "NAT Type of Destination IP (0:none, 1:inside, 2:outside)" ,
    "nat.nat_addr.src"             : "NAT Address of Source IP"                                 ,
    "nat.nat_addr.dst"             : "NAT Address of Destination IP"                            ,
    "_.bgp.nexthop_as.src"         : "BGP Next Hop ASN of Source IP"                            ,
    "_.bgp.nexthop_as.dst"         : "BGP Next Hop ASN of Destination IP"                       ,
    "_.bgp.lasthop_as.src"         : "BGP Last Hop ASN of Source IP"                            ,
    "_.bgp.lasthop_as.dst"         : "BGP Last Hop ASN of Destination IP"                       ,
    "_.bgp.thru_as_path.src"       : "BGP Through AS Path of Source IP"                         ,
    "_.bgp.thru_as_path.dst"       : "BGP Through AS Path of Destination IP"                    ,
    "_.bgp.peer_as.src"            : "BGP Peer ASN of Source IP"                                ,
    "_.bgp.peer_as.dst"            : "BGP Peer ASN of Destination IP"                           ,
    "_.bgp.as_path_len.src"        : "BGP AS Path Length of Source IP"                          ,
    "_.bgp.as_path_len.dst"        : "BGP AS Path Length of Destination IP"                     ,
    "_.bgp.thru_as_path_len.src"   : "BGP Through AS Path Length of Source IP"                  ,
    "_.bgp.thru_as_path_len.dst"   : "BGP Through AS Path Length of Destination IP"             ,
    "_.geo.longitude.src"          : "GEO Longitude of Source IP"                               ,
    "_.geo.longitude.dst"          : "GEO Longitude of Destination IP"                          ,
    "_.geo.latitude.src"           : "GEO Latitude of Source IP"                                ,
    "_.geo.latitude.dst"           : "GEO Latitude of Destination IP"                           ,
    "_.geo.country.src"            : "GEO Country Name of Source IP"                            ,
    "_.geo.country.dst"            : "GEO Country Name of Destination IP"                       ,
    "_.geo.region.src"             : "GEO Region Name of Source IP"                             ,
    "_.geo.region.dst"             : "GEO Region Name of Destination IP"                        ,
    "_.geo.city.src"               : "GEO City Name of Source IP"                               ,
    "_.geo.city.dst"               : "GEO City Name of Destination IP"                          ,
    "_.flow.duration"              : "Flow Duration in Milliseconds"                            ,
    "_.flow.dpackets"              : "Origin Packet Count"                                      ,
    "_.flow.dbytes"                : "Origin Byte Count"                                        ,
    "_.flow.dflows"                : "Origin Flow Count"                                        ,
    "_.flow.packet_rate"           : "Packet Rate"                                              ,
    "_.flow.byte_rate"             : "Byte Rate"                                                ,
    "_.flow.flow_rate"             : "Flow Rate"                                                ,
    "_.flow.avg_pkt_size"          : "Average Packet Size"                                      ,
    "_.flow.server_port.src"       : "Source Server TCP/UDP Port"                               ,
    "_.flow.server_port.dst"       : "Destination Server TCP/UDP Port"                          ,
    "_.flow.prefix.src"            : "Source IP prefix"                                         ,
    "_.flow.prefix.dst"            : "Destination IP prefix"                                    ,
    "_.dns.site_fqdn.src"          : "Full Site Name of Source IP"                              ,
    "_.dns.site_fqdn.dst"          : "Full Site Name of Destination IP"                         ,
    "_.dns.site_name.src"          : "Base Site Name of Source IP"                              ,
    "_.dns.site_name.dst"          : "Base Site Name of Destination IP"                         ,
    "_.dns.host_fqdn.src"          : "Full Host Name of Source IP"                              ,
    "_.dns.host_fqdn.dst"          : "Full Host Name of Destination IP"                         ,
    "_.dns.host_name.src"          : "Base Host Name of Source IP"                              ,
    "_.dns.host_name.dst"          : "Base Host Name of Destination IP"                         ,
    "_.dns.ott_provider.src"       : "OTT Provider of Source IP"                                ,
    "_.dns.ott_provider.dst"       : "OTT Provider of Destination IP"                           ,
    "_.dns.ott_service.src"        : "OTT Service of Source IP"                                 ,
    "_.dns.ott_service.dst"        : "OTT Service of Destination IP"                            ,
    "_.dns.cdn_provider.src"       : "CDN Provider of Source IP"                                ,
    "_.dns.cdn_provider.dst"       : "CDN Provider of Destination IP"                           ,
    "_.nat.inside_addr.src"        : "NAT inside address of Source IP"                          ,
    "_.nat.inside_addr.dst"        : "NAT inside address of Destination IP"                     ,
    "_.nat.outside_addr.src"       : "NAT outside address of Source IP"                         ,
    "_.nat.outside_addr.dst"       : "NAT outside address of Destination IP"                    ,
    "_.nat.inside_port.src"        : "NAT inside port of Source Port"                           ,
    "_.nat.inside_port.dst"        : "NAT inside port of Destination Port"                      ,
    "_.nat.outside_port.src"       : "NAT outside port of Source Port"                          ,
    "_.nat.outside_port.dst"       : "NAT outside port of Destination Port"                     ,
    "dns.site_fqdn.src"            : "Site FQDN of Source"                                      ,
    "dns.site_fqdn.dst"            : "Site FQDN of Destination"                                 ,
    "dns.host_fqdn.src"            : "Host FQDN of Source"                                      ,
    "dns.host_fqdn.dst"            : "Host FQDN of Destination"                                 ,
}

field_table = FieldTable().load(DefaultConf)