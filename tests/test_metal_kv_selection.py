"""Compile the actual Metal selector and cover its supported cache/shape domains."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
METAL = ROOT / 'llama-cpp-turboquant/ggml/src/ggml-metal'


def test_mixed_and_turbo_kv_select_existing_baseline_variants(tmp_path):
    compiler = shutil.which('c++') or shutil.which('g++')
    if not compiler:
        pytest.skip('C++ compiler unavailable')
    source = tmp_path / 'selection.cpp'
    source.write_text(r'''
#include "ggml-metal-tuning.h"
#include <cassert>
#include <cstdio>
using namespace ggml_metal_tuning;
int main() {
    const int types[] = {GGML_TYPE_Q8_0,GGML_TYPE_TURBO2_0,GGML_TYPE_TURBO3_0,GGML_TYPE_TURBO4_0};
    const int heads[][2] = {{32,32},{64,64},{96,96},{128,128},{192,192},{192,128},{256,256},{320,256},{512,512},{576,512}};
    int count = 0;
    for (int family : {7,8,9,10}) for (int key : types) for (int value : types) for (const auto & head : heads)
    for (int past : {512,1024,4096,8192,16384,65536}) for (int query : {1,2,4,16}) {
        if (key == GGML_TYPE_Q8_0 && value == GGML_TYPE_Q8_0) continue;
        const auto expected = fa_vec_baseline_cfg(head[0],head[1]);
        const auto actual = fa_vec_pick_kv(family,key,value,head[0],head[1],past,query);
        assert(actual.Q == expected.Q && actual.NE == expected.NE);
        count++;
    }
    // Standard symmetric caches retain their tuned choices.
    for (int family : {7,8,9,10}) for (int dtype : {GGML_TYPE_F16,GGML_TYPE_BF16,GGML_TYPE_Q4_0,GGML_TYPE_Q4_1,GGML_TYPE_Q5_0,GGML_TYPE_Q5_1,GGML_TYPE_Q8_0})
    for (int past : {512,1024,4096,8192,16384,65536}) {
        const auto expected=fa_vec_pick(family,dtype,256,256,past,1);
        const auto actual=fa_vec_pick_kv(family,dtype,dtype,256,256,past,1);
        assert(actual.Q == expected.Q && actual.NE == expected.NE);
    }
    // A tuning override cannot request uncompiled TurboQuant variants.
    fa_vec_set_override({4,4});
    const auto actual=fa_vec_pick_kv(9,GGML_TYPE_Q8_0,GGML_TYPE_TURBO4_0,256,256,8192,1);
    assert(actual.Q == 1 && actual.NE == fa_vec_baseline_ne(256,256));
    fa_vec_clear_override();
    std::printf("%d cache/shape/family cases passed\n",count);
}
''')
    binary = tmp_path / 'selection'
    subprocess.run([compiler,'-std=c++17','-I'+str(METAL),
                    '-I'+str(ROOT/'llama-cpp-turboquant/ggml/include'),
                    str(source),str(METAL/'ggml-metal-tuning.cpp'),'-o',str(binary)],
                   check=True,capture_output=True,text=True,timeout=90)
    result = subprocess.check_output([str(binary)],text=True,timeout=30)
    assert '14400 cache/shape/family cases passed' in result
    shader = (METAL / 'kernels/fa_vec_turbo.metal').read_text()
    # The selected names correspond to supplied templates, rather than tuned suffixes.
    types = ('q8_0','turbo2','turbo3','turbo4')
    for key in types:
        for value in types:
            if key == value == 'q8_0':
                continue
            pair = key if key == value else f'k{key}_v{value}'
            for dk,dv in ((32,32),(64,64),(96,96),(128,128),(192,192),(192,128),(256,256),(320,256),(512,512),(576,512)):
                assert f'kernel_flash_attn_ext_vec_{pair}_dk{dk}_dv{dv}' in shader
