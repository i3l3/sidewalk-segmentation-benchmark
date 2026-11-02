import os
import unittest
from unittest.mock import patch, Mock
from src.aihub_download import proxy_environment
from src import tailscale_proxy as ts

class ProxyTests(unittest.TestCase):
    def test_aihub_proxy_is_scoped(self):
        with patch.dict(os.environ, {'AIHUB_PROXY_URL':ts.PROXY,'HTTPS_PROXY':'http://original','NO_PROXY':'*'}):
            env=proxy_environment()
            self.assertEqual(env['HTTPS_PROXY'],ts.PROXY)
            self.assertEqual(env['https_proxy'],ts.PROXY)
            self.assertEqual(env['NO_PROXY'],'')
            self.assertEqual(os.environ['HTTPS_PROXY'],'http://original')

    def test_reject_nonlocal_or_authenticated_proxy(self):
        for proxy in ['http://remote:123','http://user:pass@localhost:123','socks5://localhost:123']:
            with patch.dict(os.environ,{'AIHUB_PROXY_URL':proxy}):
                with self.assertRaises(ValueError):proxy_environment()

    def test_select_rejects_offline_node_and_clears_proxy(self):
        with patch.dict(os.environ,{'AIHUB_PROXY_URL':ts.PROXY}), patch.object(ts,'exit_nodes',return_value=[{'name':'home','ip':'100.1.1.1','dns_name':'home.example.','online':False}]):
            with self.assertRaises(ValueError):ts.select_exit_node('home')
            self.assertNotIn('AIHUB_PROXY_URL',os.environ)

    def test_failed_proxy_probe_does_not_enable_aihub_proxy(self):
        nodes=[{'name':'home','ip':'100.1.1.1','dns_name':'home.example.','online':True}]
        with patch.dict(os.environ,{},clear=True), patch.object(ts,'exit_nodes',return_value=nodes), patch.object(ts,'_cli',return_value=Mock(returncode=0)), patch.object(ts,'status',return_value={'ExitNodeStatus':{'Online':True}}), patch.object(ts.urllib.request,'build_opener',side_effect=ConnectionError):
            with self.assertRaises(ConnectionError):ts.select_exit_node('home')
            self.assertNotIn('AIHUB_PROXY_URL',os.environ)

    def test_proxy_is_enabled_only_after_probe(self):
        nodes=[{'name':'home','ip':'100.1.1.1','dns_name':'home.example.','online':True}]
        response=Mock();response.read.return_value=b'{"ip":"203.0.113.1"}'
        opener=Mock();opener.open.return_value.__enter__=Mock(return_value=response);opener.open.return_value.__exit__=Mock(return_value=False)
        with patch.dict(os.environ,{},clear=True), patch.object(ts,'exit_nodes',return_value=nodes), patch.object(ts,'_cli',return_value=Mock(returncode=0)), patch.object(ts,'status',return_value={'ExitNodeStatus':{'Online':True}}), patch.object(ts.urllib.request,'build_opener',return_value=opener):
            self.assertEqual(ts.select_exit_node('home'),'203.0.113.1')
            self.assertEqual(os.environ['AIHUB_PROXY_URL'],ts.PROXY)

if __name__=='__main__':unittest.main()
