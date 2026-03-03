import glfw
import moderngl
import numpy as np
import time
from PIL import Image
from datetime import datetime

WINDOW_W, WINDOW_H = 1280, 720
BENCH_W, BENCH_H = 2048, 2048
BENCH_ITERATIONS = 3000

VERTEX_SHADER = '''
#version 330
in vec2 in_vert;
out vec2 v_uv;
void main() {
    v_uv = in_vert * 0.5 + 0.5;
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
'''

BUFFER_A_FRAG = '''
#version 330
out vec4 fragColor;
in vec2 v_uv;
uniform vec3  iResolution;
uniform float iTime;

vec2 hash22(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)),
             dot(p, vec2(269.5, 183.3)));
    return fract(sin(p) * 43758.5453);
}

float worley(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    float minDist = 1e9;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 neighbor = vec2(float(x), float(y));
            vec2 point = hash22(i + neighbor);
            point = 0.5 + 0.5 * sin(iTime * 0.3 + 6.2831 * point);
            vec2 diff = neighbor + point - f;
            minDist = min(minDist, dot(diff, diff));
        }
    }
    return clamp(sqrt(minDist), 0.0, 1.0);
}

void main() {
    vec2 fragCoord = v_uv * iResolution.xy;
    vec2 uv = (fragCoord / iResolution.xy) * 8.0;
    float n = worley(uv);
    fragColor = vec4(n, n, n, 1.0);
}
'''

MAIN_IMAGE_FRAG = '''
#version 330
out vec4 fragColor;
in vec2 v_uv;

uniform vec3      iResolution;
uniform float     iTime;
uniform vec4      iMouse;
uniform sampler2D iChannel0;

const int   MAX_ITER = 30;
const float ALPHA    = 0.6;

vec2 cMul(vec2 a, vec2 b) { return vec2(a.x*b.x - a.y*b.y,  a.x*b.y + a.y*b.x); }
vec2 cDiv(vec2 a, vec2 b) { float d = dot(b, b) + 1e-9; return vec2(dot(a, b),  a.y*b.x - a.x*b.y) / d; }

vec3 pinkColor(float h) {
    vec3 c1 = vec3(0.55, 0.05, 0.20);
    vec3 c2 = vec3(0.88, 0.30, 0.48);
    vec3 c3 = vec3(0.97, 0.68, 0.75);
    vec3 c4 = vec3(1.00, 0.94, 0.95);
    if      (h < 0.3) return mix(c1, c2, h / 0.3);
    else if (h < 0.7) return mix(c2, c3, (h - 0.3) / 0.4);
    else              return mix(c3, c4, (h - 0.7) / 0.3);
}

float getHeight(vec2 c, vec2 z, vec2 p1) {
    float kConverged = float(MAX_ITER);
    vec2 omega = vec2(-0.5,  0.8660254);
    vec2 one   = vec2( 1.0,  0.0);
    vec2 tlc = vec2(1.0 + 2.0*p1.x,   1.7320508 + 2.0*p1.y);
    vec2 tc  = vec2((p1.x - 1.0 - 1.7320508*p1.y) * 0.5, (p1.y + 1.7320508*(p1.x + 1.0)) * 0.5);
    for (int i = 0; i < MAX_ITER; i++) {
        vec2 oldZ = z;
        vec2 uvNorm = fract(z / 16.0 + 0.5);
        float n     = texture(iChannel0, uvNorm).r;
        vec2 zd  = z + n * ALPHA;
        vec2 pz  = cMul(cMul(zd - p1, zd - one), zd - omega);
        vec2 z2  = cMul(zd, zd);
        vec2 dpz = 3.0*z2 - cMul(tlc, zd) + tc;
        z = zd - cDiv(pz, dpz) + c;
        if (dot(z - oldZ, z - oldZ) < 0.000025) {
            kConverged = float(i);
            break;
        }
    }
    return (float(MAX_ITER) - kConverged) / float(MAX_ITER);
}

void main() {
    vec2 fragCoord = v_uv * iResolution.xy;
    float t = iTime;
    vec2 camPos = vec2(sin(t*0.1), cos(t*0.1)) * 0.2;
    vec2 uv     = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    uv          = uv * 2.5 + camPos;
    vec2 p1     = vec2(cos(t), sin(t)) * 0.8;
    if (iMouse.z > 0.0) {
        p1 = (iMouse.xy - 0.5 * iResolution.xy) / iResolution.y * 2.5 + camPos;
    }
    vec2 zInit  = vec2(sin(t*0.3)*0.5, -0.2);
    float h     = getHeight(uv, zInit, p1);
    h = 1.0 - h;
    vec3 col    = pinkColor(h) * (1.0 + 0.01*h);
    vec2 uvN    = fragCoord / iResolution.xy - 0.5;
    float vig   = 1.0 - dot(uvN, uvN) * 1.2;
    fragColor   = vec4(col * vig, 1.0);
}
'''

class RenderApp:
    def __init__(self):
        if not glfw.init():
            raise Exception("GLFW Init Failed")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, False)

        self.window = glfw.create_window(WINDOW_W, WINDOW_H, "NINF Terrain - Paper Benchmark", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)

        self.ctx = moderngl.create_context()

        self.vbo = self.ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype='f4').tobytes())

        self.prog_noise = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=BUFFER_A_FRAG)
        self.prog_main  = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=MAIN_IMAGE_FRAG)

        self.vao_noise = self.ctx.simple_vertex_array(self.prog_noise, self.vbo, 'in_vert')
        self.vao_main  = self.ctx.simple_vertex_array(self.prog_main,  self.vbo, 'in_vert')

        self.setup_noise_texture()
        self.setup_benchmark_fbo()

        self.start_time    = time.time()
        self.render_enabled = True
        self.mouse_pos     = [0, 0, 0, 0]

        glfw.set_key_callback(self.window,          self.key_callback)
        glfw.set_cursor_pos_callback(self.window,   self.cursor_callback)
        glfw.set_mouse_button_callback(self.window, self.mouse_callback)

    def setup_noise_texture(self):
        size = (BENCH_W, BENCH_H)
        self.noise_tex = self.ctx.texture(size, 4, dtype='f4')
        self.noise_tex.repeat_x = True
        self.noise_tex.repeat_y = True
        self.noise_tex.filter   = (moderngl.LINEAR, moderngl.LINEAR)

        fbo = self.ctx.framebuffer(color_attachments=[self.noise_tex])
        fbo.use()
        self.prog_noise['iResolution'].value = (size[0], size[1], 1.0)
        self.prog_noise['iTime'].value = 0.0
        self.vao_noise.render(moderngl.TRIANGLE_STRIP)
        self.ctx.screen.use()
        print(f"[System] Noise Buffer Precomputed ({size[0]}x{size[1]})")

    def setup_benchmark_fbo(self):
        self.bench_tex = self.ctx.texture((BENCH_W, BENCH_H), 4, dtype='f4')
        self.bench_fbo = self.ctx.framebuffer(color_attachments=[self.bench_tex])

    def save_screenshot(self):
        t = time.time() - self.start_time

        self.bench_fbo.use()
        self.bench_fbo.viewport = (0, 0, BENCH_W, BENCH_H)

        self.noise_tex.use(location=0)
        self.prog_main['iChannel0'].value  = 0
        self.prog_main['iResolution'].value = (BENCH_W, BENCH_H, 1.0)
        self.prog_main['iTime'].value       = t
        self.prog_main['iMouse'].value      = tuple(self.mouse_pos)

        self.vao_main.render(moderngl.TRIANGLE_STRIP)
        self.ctx.finish()

        raw = self.bench_fbo.read(components=4, dtype='f4')
        arr = np.frombuffer(raw, dtype=np.float32).reshape(BENCH_H, BENCH_W, 4)
        arr = np.flipud(arr)
        arr = np.clip(arr[:, :, :3] * 255, 0, 255).astype(np.uint8)

        filename = datetime.now().strftime("screenshot_%Y%m%d_%H%M%S.png")
        Image.fromarray(arr, 'RGB').save(filename)
        print(f"[Screenshot] Saved: {filename}  ({BENCH_W}x{BENCH_H})")

        self.ctx.screen.use()
        self.ctx.screen.viewport = (0, 0, WINDOW_W, WINDOW_H)
        self.prog_main['iResolution'].value = (WINDOW_W, WINDOW_H, 1.0)

    def run_benchmark(self):
        print(f"\n{'='*60}")
        print(f" STARTING BENCHMARK: {BENCH_ITERATIONS} Frames @ {BENCH_W}x{BENCH_H}")
        print(f"{'='*60}")

        self.bench_fbo.use()
        self.bench_fbo.viewport = (0, 0, BENCH_W, BENCH_H)

        self.noise_tex.use(location=0)
        self.prog_main['iChannel0'].value   = 0
        self.prog_main['iResolution'].value = (BENCH_W, BENCH_H, 1.0)
        self.prog_main['iMouse'].value      = (0, 0, 0, 0)

        print("Warmup GPU (100 frames)...")
        for _ in range(100):
            self.prog_main['iTime'].value = 1.0
            self.vao_main.render(moderngl.TRIANGLE_STRIP)
        self.ctx.finish()

        print("Running benchmark...")
        t_start = time.perf_counter()
        for i in range(BENCH_ITERATIONS):
            self.prog_main['iTime'].value = i * 0.01
            self.vao_main.render(moderngl.TRIANGLE_STRIP)
        self.ctx.finish()
        t_end = time.perf_counter()

        duration = t_end - t_start
        avg_fps  = BENCH_ITERATIONS / duration
        avg_ms   = (duration / BENCH_ITERATIONS) * 1000.0

        print(f"{'-'*60}")
        print(f" RESULTS:")
        print(f" Resolution : {BENCH_W} x {BENCH_H}")
        print(f" Total Time : {duration:.4f} s")
        print(f" Avg FPS    : {avg_fps:.2f}")
        print(f" Avg Latency: {avg_ms:.4f} ms")
        print(f"{'='*60}\n")

        self.ctx.screen.use()
        self.ctx.screen.viewport = (0, 0, WINDOW_W, WINDOW_H)
        self.prog_main['iResolution'].value = (WINDOW_W, WINDOW_H, 1.0)

    def key_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_B:
                self.run_benchmark()
            elif key == glfw.KEY_S:
                self.save_screenshot()
            elif key == glfw.KEY_V:
                self.render_enabled = not self.render_enabled
                print(f"[System] Visual Rendering: {'ON' if self.render_enabled else 'OFF'}")
            elif key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)

    def cursor_callback(self, window, x, y):
        self.mouse_pos[0] = x
        self.mouse_pos[1] = WINDOW_H - y

    def mouse_callback(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.mouse_pos[2] = 1.0 if action == glfw.PRESS else 0.0

    def run(self):
        print(
            f"Controls:\n"
            f" [B]   Run Benchmark ({BENCH_ITERATIONS} frames @ {BENCH_W}x{BENCH_H})\n"
            f" [S]   Save Screenshot ({BENCH_W}x{BENCH_H} PNG)\n"
            f" [V]   Toggle Visual Rendering\n"
            f" [Esc] Quit"
        )

        while not glfw.window_should_close(self.window):
            if self.render_enabled:
                t = time.time() - self.start_time
                self.ctx.clear()
                self.noise_tex.use(location=0)

                self.prog_main['iTime'].value   = t
                self.prog_main['iMouse'].value  = tuple(self.mouse_pos)
                self.prog_main['iChannel0'].value = 0
                self.prog_main['iResolution'].value = (WINDOW_W, WINDOW_H, 1.0)

                self.vao_main.render(moderngl.TRIANGLE_STRIP)
                glfw.swap_buffers(self.window)
            else:
                time.sleep(0.1)

            glfw.poll_events()

        glfw.terminate()


if __name__ == '__main__':
    app = RenderApp()
    app.run()