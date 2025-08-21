local ffi = require("ffi")
local c_entity = require("gamesense/entity")
local pui = require("gamesense/pui")
local base64 = require("gamesense/base64")
local clipboard = require("gamesense/clipboard")
local vector = require("vector")
local json = require("json")
local csgo_weapons = require('gamesense/csgo_weapons')
local chat = require("gamesense/chat")
local localize = require("gamesense/localize")
local antiaim_funcs = require("gamesense/antiaim_funcs")
local websockets = require ("gamesense/websockets")
local http = require('gamesense/http')
local trace = require 'gamesense/trace'

local dragging = (function() local a={}local b,c,d,e,f,g,h,i,j,k,l,m,n,o;local p={__index={drag=function(self,...)local q,r=self:get()local s,t=a.drag(q,r,...)if q~=s or r~=t then self:set(s,t)end;return s,t end,set=function(self,q,r)local j,k=client.screen_size()ui.set(self.x_reference,q/j*self.res)ui.set(self.y_reference,r/k*self.res)end,get=function(self)local j,k=client.screen_size()return ui.get(self.x_reference)/self.res*j,ui.get(self.y_reference)/self.res*k end}}function a.new(u,v,w,x)x=x or 10000;local j,k=client.screen_size()local y=ui.new_slider("LUA","A",u.." window position",0,x,v/j*x)local z=ui.new_slider("LUA","A","\n"..u.." window position y",0,x,w/k*x)ui.set_visible(y,false)ui.set_visible(z,false)return setmetatable({name=u,x_reference=y,y_reference=z,res=x},p)end;client.set_event_callback("paint",function()c=ui.is_menu_open()f,g=d,e;d,e=ui.mouse_position()i=h;h=client.key_state(0x01)==true;m=l;l={}o=n;n=false;j,k=client.screen_size()end)function a.drag(q,r,A,B,C,D,E)if c and i~=nil then if(not i or o)and h and f>q and g>r and f<q+A and g<r+B then n=true;q,r=q+d-f,r+e-g;if not D then q=math.max(0,math.min(j-A,q))r=math.max(0,math.min(k-B,r))end;if E then end end end;table.insert(l,{q,r,A,B})return q,r,A,B end;return a end)()

local username = _USER_NAME and _USER_NAME or 'Lopgy'
local build = 'pre-release'

local ffi_helpers do
    ffi_helpers = {} do
        ffi_helpers.get_client_entity = vtable_bind('client.dll', 'VClientEntityList003', 3, 'void*(__thiscall*)(void***, int)')

        ffi_helpers.animlayers = {} do
            if not pcall(ffi.typeof, 'bt_animlayer_t') then
                ffi.cdef[[
                    typedef struct {
                        float   anim_time;
                        float   fade_out_time;
                        int     nil;
                        int     activty;
                        int     priority;
                        int     order;
                        int     sequence;
                        float   prev_cycle;
                        float   weight;
                        float   weight_delta_rate;
                        float   playback_rate;
                        float   cycle;
                        int     owner;
                        int     bits;
                    } bt_animlayer_t, *pbt_animlayer_t
                ]]
            end

            ffi_helpers.animlayers.offset = ffi.cast('int*', ffi.cast('uintptr_t', client.find_signature('client.dll', '\x8B\x89\xCC\xCC\xCC\xCC\x8D\x0C\xD1')) + 2)[0]

            ffi_helpers.animlayers.get = function (self, ent)
                local client_entity = ffi_helpers.get_client_entity(ent)

                if not client_entity then
                    return
                end

                return ffi.cast('pbt_animlayer_t*', ffi.cast('uintptr_t', client_entity) + self.offset)[0]
            end
        end

        ffi_helpers.activity = {} do
            if not pcall(ffi.typeof, 'bt_get_sequence') then
                ffi.cdef[[
                    typedef int(__fastcall* bt_get_sequence)(void* entity, void* studio_hdr, int sequence);
                ]]
            end

            ffi_helpers.activity.offset = 0x2950
            ffi_helpers.activity.location = ffi.cast('bt_get_sequence', client.find_signature('client.dll', '\x55\x8B\xEC\x53\x8B\x5D\x08\x56\x8B\xF1\x83'))

            ffi_helpers.activity.get = function (self, sequence, ent)
                if ent == nil then return end
                local client_entity = ffi_helpers.get_client_entity(ent)

                if not client_entity then
                    return
                end

                local studio_hdr = ffi.cast('void**', ffi.cast('uintptr_t', client_entity) + self.offset)[0]

                if not studio_hdr then
                    return;
                end

                return self.location(client_entity, studio_hdr, sequence);
            end
        end
    end
end

local override = {} do
    local e_hotkey_mode = {
        [0] = "Always on",
        [1] = "On hotkey",
        [2] = "Toggle",
        [3] = "Off hotkey"
    }

    local data = { }

    local function get_value(ref)
        local value = { ui.get(ref) }
        local typeof = ui.type(ref)

        if typeof == "hotkey" then
            return { e_hotkey_mode[value[2]], value[3] }
        end

        return value
    end

    function override.get(ref, ...)
        local value = data[ref]

        if value == nil then
            return
        end

        return unpack(value)
    end

    function override.set(ref, ...)
        if data[ref] == nil then
            data[ref] = get_value(ref)
        end

        ui.set(ref, ...)
    end

    function override.unset(ref)
        if data[ref] == nil then
            return
        end

        ui.set(ref, unpack(data[ref]))
        data[ref] = nil
    end
end

local lua_db = {
    config = 'ayesobaka',
    user = 'Asaka_lua:user'
}

local user_info = database.read(lua_db.user) or {0, 0, 0}

local start_time = client.unix_time()

local function get_elapsed_time(zalupa)
    local hours = math.floor(zalupa / 3600)
    local minutes = math.floor((zalupa - hours * 3600) / 60)
    local seconds = math.floor(zalupa - hours * 3600 - minutes * 60)
    return string.format("%02d:%02d:%02d", hours, minutes, seconds)
end

local lua_items = {}
local aa_helpers = {}
local misc_helpers = {}
local visuals_helpers = {}
local rage_helpers = {}


local ref = {
    enabled = ui.reference('AA', 'Anti-aimbot angles', 'Enabled'),
    yawbase = ui.reference('AA', 'Anti-aimbot angles', 'Yaw base'),
    fsbodyyaw = ui.reference('AA', 'anti-aimbot angles', 'Freestanding body yaw'),
    edgeyaw = ui.reference('AA', 'Anti-aimbot angles', 'Edge yaw'),
    fakeduck = ui.reference('RAGE', 'Other', 'Duck peek assist'),
    forcebaim = ui.reference('RAGE', 'Aimbot', 'Force body aim'),
    safepoint = ui.reference('RAGE', 'Aimbot', 'Force safe point'),
    roll = { ui.reference('AA', 'Anti-aimbot angles', 'Roll') },
    clantag = ui.reference('Misc', 'Miscellaneous', 'Clan tag spammer'),
    legs = ui.reference('AA', 'Other', 'Leg Movement'),

    pitch = { ui.reference('AA', 'Anti-aimbot angles', 'pitch'), },
    rage = { ui.reference('RAGE', 'Aimbot', 'Enabled') },
    yaw = { ui.reference('AA', 'Anti-aimbot angles', 'Yaw') }, 
    yawjitter = { ui.reference('AA', 'Anti-aimbot angles', 'Yaw jitter') },
    bodyyaw = { ui.reference('AA', 'Anti-aimbot angles', 'Body yaw') },
    freestand = { ui.reference('AA', 'Anti-aimbot angles', 'Freestanding') },
    slow = { ui.reference('AA', 'Other', 'Slow motion') },
    os = { ui.reference('AA', 'Other', 'On shot anti-aim') },
    dt = { ui.reference('RAGE', 'Aimbot', 'Double tap') },
    minimum_damage = ui.reference("RAGE", "Aimbot", "Minimum damage"),
    minimum_damage_override = { ui.reference("RAGE", "Aimbot", "Minimum damage override") },
    quick_peek = { ui.reference('RAGE', 'Other', 'Quick peek assist') },

    aimbot = ui.reference('RAGE', 'Aimbot', 'Enabled'),
    doubletap = {
        main = { ui.reference('RAGE', 'Aimbot', 'Double tap') },
        fakelag_limit = ui.reference('RAGE', 'Aimbot', 'Double tap fake lag limit'),
    },
    peek = { ui.reference('RAGE', 'Other', 'Quick peek assist') },
    fov = ui.reference("Misc", "Miscellaneous", "Override FOV"),
    fakelag = {ui.reference('AA', 'Fake lag', 'Enabled')},
    fakelag_amount = ui.reference('AA', 'Fake lag', 'Amount'),  -- 'Dynamic', 'Maximum', 'Fluctuate'
    fakelag_variance = ui.reference('AA', 'Fake lag', 'Variance'), --0, 100%
    fakelag_limit = ui.reference('AA', 'Fake lag', 'Limit'), --1, 15t
    scope = ui.reference('VISUALS', 'Effects', 'Remove scope overlay'),

    slowmotion = {ui.reference('AA', 'Other', 'Slow motion')},
    legmovement = ui.reference('AA', 'Other', 'Leg movement'),
    flickpeek = {ui.reference('AA', 'Other', 'Fake peek')},
}

math.clamp = function (x, a, b)
    if a > x then return a
    elseif b < x then return b
    else return x end
end

math.lerping = function (a, b, w)
    return a + (b - a) * w
end

math.lerp = function (start, enp, time)
    time = time or 0.005
    time = math.clamp(globals.absoluteframetime() * time * 175.0, 0.01, 1.0)
    local a = math.lerping(start, enp, time)
    if enp == 0.0 and a < 0.02 and a > -0.02 then
        a = 0.0
    elseif enp == 1.0 and a < 1.01 and a > 0.99 then
        a = 1.0
    end
    return a
end

function math.normalize(x, min, max)
    local d = max - min

    while x < min do
        x = x + d
    end

    while x > max do
        x = x - d
    end

    return x
end

renderer.rec = function(x, y, w, h, radius, color)
    radius = math.min(x/2, y/2, radius)
    local r, g, b, a = unpack(color)
    renderer.rectangle(x, y + radius, w, h - radius*2, r, g, b, a)
    renderer.rectangle(x + radius, y, w - radius*2, radius, r, g, b, a)
    renderer.rectangle(x + radius, y + h - radius, w - radius*2, radius, r, g, b, a)
    renderer.circle(x + radius, y + radius, r, g, b, a, radius, 180, 0.25)
    renderer.circle(x - radius + w, y + radius, r, g, b, a, radius, 90, 0.25)
    renderer.circle(x - radius + w, y - radius + h, r, g, b, a, radius, 0, 0.25)
    renderer.circle(x + radius, y - radius + h, r, g, b, a, radius, -90, 0.25)
end

renderer.rec_outline = function(x, y, w, h, radius, thickness, color)
    radius = math.min(w/2, h/2, radius)
    local r, g, b, a = unpack(color)
    if radius == 1 then
            renderer.rectangle(x, y, w, thickness, r, g, b, a)
            renderer.rectangle(x, y + h - thickness, w , thickness, r, g, b, a)
    else
        renderer.rectangle(x + radius, y, w - radius*2, thickness, r, g, b, a)
        renderer.rectangle(x + radius, y + h - thickness, w - radius*2, thickness, r, g, b, a)
        renderer.rectangle(x, y + radius, thickness, h - radius*2, r, g, b, a)
        renderer.rectangle(x + w - thickness, y + radius, thickness, h - radius*2, r, g, b, a)
        renderer.circle_outline(x + radius, y + radius, r, g, b, a, radius, 180, 0.25, thickness)
        renderer.circle_outline(x + radius, y + h - radius, r, g, b, a, radius, 90, 0.25, thickness)
        renderer.circle_outline(x + w - radius, y + radius, r, g, b, a, radius, -90, 0.25, thickness)
        renderer.circle_outline(x + w - radius, y + h - radius, r, g, b, a, radius, 0, 0.25, thickness)
    end
end

renderer.glow_module = function(x, y, w, h, width, rounding, accent, accent_inner)
    local thickness = 1
    local offset = 1
    local r, g, b, a = unpack(accent)
    if accent_inner then
        renderer.rec(x , y, w, h + 1, rounding, accent_inner)
    end
    for k = 0, width do
        if a * (k/width)^(1) > 5 then
            local accent = {r, g, b, a * (k/width)^(2)}
            renderer.rec_outline(x + (k - width - offset)*thickness, y + (k - width - offset) * thickness, w - (k - width - offset)*thickness*2, h + 1 - (k - width - offset)*thickness*2, rounding + thickness * (width - k + offset), thickness, accent)
        end
    end
end

local lua_group = pui.group("aa", "anti-aimbot angles")
local fakelag_group = pui.group("aa", "Fake lag")
local other_group = pui.group("aa", "Other")

local aa_conditions = {'Global', 'Stand', 'Move', 'Walk', 'Crouch', 'Crouch+Move', 'Air', 'Air+C', 'Discharged', 'Legit AA'}

lua_items.main = {
    label1 = fakelag_group:label('          Asaka ~ 2025 '),
    labelsp = fakelag_group:label(' '),
    labeluser = fakelag_group:label(' \v\r  Welcome Back, \v'..username),
    labelonline = fakelag_group:label(' \v\r  Online Users: \v25'),
    labeltime = fakelag_group:label(' \v\r  Time Played: \v'..get_elapsed_time(user_info[3])),
    labelmiss = fakelag_group:label(' \v\r  Missed Shots At You: \v'..user_info[2]),
    labelenem = fakelag_group:label(' \v\r  Enemies Eliminated: \v'..user_info[1] or '0'),
    label2 = fakelag_group:label(' '),
    lol = lua_group:checkbox('#is_open'),
    tab = lua_group:combobox('lua tab', {'Anti-Aim', 'Builder', 'Ragebot', 'Misc', 'Visuals', 'Configs'}),
    back = lua_group:button('\v\r  Back', function() end),
    labelaa = lua_group:label(' '),
    antiaim = lua_group:button('\v\r  Anti-Aim', function() end),
    labelrg = lua_group:label(' '),
    ragebot = lua_group:button('\v\r  Ragebot', function() end),
    labelvs = lua_group:label(' '),
    visuals = lua_group:button('\v\r    Visuals', function() end),
    labelmc = lua_group:label(' '),
    misc = lua_group:button('\v\r        Misc', function() end),
    labelcf = lua_group:label(' '),
    configs = lua_group:button('\v\r   Configs', function() end),
    labells = lua_group:label(' '),
    label3 = lua_group:label(' '),

    label_1 = other_group:label(' '),
    discord = other_group:button('\v\r  Rollmops Discord Server', function() 
        panorama.open().SteamOverlayAPI.OpenExternalBrowserURL("https://discord.gg/5A86H2n6NE")
    end),
    label_2 = other_group:label(' '),
    league = other_group:button('\v\r   lectorz Discord Server', function() 
        panorama.open().SteamOverlayAPI.OpenExternalBrowserURL("https://discord.gg/mYpaBM5e3Q")
    end),
    label_3 = other_group:label(' '),
    yt = other_group:button('\v\r   Youtube', function() 
        panorama.open().SteamOverlayAPI.OpenExternalBrowserURL("https://www.youtube.com/Lopgy233")
    end),
}

lua_items.antiaim = {
    tab = fakelag_group:combobox('#tab', {'Main', 'Keybinds', 'Fake Lag'}),
    main = fakelag_group:button('\v\r  Main', function() end),
    keybinds = fakelag_group:button('\v\r  Keybinds', function() end),
    fakelag = fakelag_group:button('\v\r  Fake Lag', function() end),
    addons = fakelag_group:multiselect('\v\r  Additions', {'Override Anti-Aim', 'Anti Backstab', 'Safe Head'}),
    aa_override = fakelag_group:multiselect('\v\r  Override When', {'No Enemies Alive', 'Warmup'}),
    safe_head = fakelag_group:multiselect('\v\r  Safe Head', {'Air+C Knife', 'Air+C Zeus', 'Air+C SMG', 'Height Difference', 'Stand', 'Air', 'Air+C', 'Crouch', 'Crouch Move', 'Fake Duck', 'Fake Lags'}),
    height_difference = fakelag_group:slider('\v\r  Difference', 0, 300, 200, true, '%'),
    defensive_triggers = fakelag_group:multiselect('\v\r  Force Defensive Triggers', {'Hittable', 'Reload', 'Flashed', 'Damaged', 'Weapon Swap'}),
    defensive_condition = fakelag_group:multiselect('\v\r  Conditions', {'Stand', 'Moving', 'Walking', 'Crouching', 'Air'}),
    fl_custom = fakelag_group:checkbox('\v\r  Custom Fakelag'),
    fl_c_type = fakelag_group:combobox('\v\r  Fakelag Type ', {'Cycle', 'Random'}),
    fl_c_limit = fakelag_group:slider('\v\r  Fakelag Limit ', 1, 15, 14, true, 't'),
    fl_amount = fakelag_group:combobox('\v\r  Fakelag Type', {'Dynamic', 'Maximum', 'Fluctuate'}),
    fl_variance = fakelag_group:slider('\v\r  Fakelag Variance', 0, 100, 0, true, '%'),
    fl_limit = fakelag_group:slider('\v\r  Fakelag Limit', 1, 15, 14, true, 't'),
    fl_disablers = fakelag_group:multiselect('\v\r  Fakelag Disablers', {'Crouch', 'Stand'}),

    edgeyaw_on_fd = fakelag_group:checkbox('\v\r  Edge Yaw On Fake Duck'),
    freestanding = fakelag_group:checkbox('\v⭿\r  Freestanding'),
    freestanding_key = fakelag_group:hotkey('\v⭿\r  Freestanding', true),
    fr_options = fakelag_group:multiselect('\v\r  Options', {'On Quick Peek', 'Disablers', 'Disable Yaw Modifier', 'Flick peek'}),
    fr_disablers = fakelag_group:multiselect('\v\r  Disablers', {'Walking', 'Crouch', 'Air'}),
    manual_keybinds = fakelag_group:checkbox('\v\r  Manual Keybinds'),
    yaw_options = fakelag_group:multiselect('\v\r  Keybinds Options', {'Disable Yaw Modifier', 'Flick peek'}),
    key_left = fakelag_group:hotkey('\v\r  Left Manual'),
    key_right = fakelag_group:hotkey('\v\r  Right Manual'),
    key_forward = fakelag_group:hotkey('\v\r  Forward Manual'),

    is_def = lua_group:checkbox('#defensive'),
    to_def = lua_group:button('\v\r  Defensive', function() end),
    to_build = lua_group:button('\v\r  Builder', function() end),
    label1 = lua_group:label(' '),

    pitch = lua_group:combobox('\v\r  Pitch', {"Off", "Down", 'Random'}),
    yaw_base = lua_group:combobox('\v\r  Yaw Base', {'Local view', 'At targets'}),
    condition = lua_group:combobox('\v\r  Condition', aa_conditions),
    label2 = lua_group:label(' '),
}

lua_items.ragebot = {
    predict = lua_group:checkbox('\v\r  Predict'),
    predict_key = lua_group:hotkey('\v\r  Predict', true),
    pr_over_clor = lua_group:checkbox('\v\r  Disable clock drifting'),
    label1 = lua_group:label(' '),
    defensive_fix = lua_group:checkbox('\v\r  Defensive Fix'),
    hideshot_fix = lua_group:checkbox('\v\r  HideShots Fix'),
    label2 = lua_group:label(' '),
    resolver = lua_group:checkbox('\v\r  Jitter Resolver'),
    def_resovler = lua_group:checkbox('\v\r  Defensive Resolver'),
    label3 = lua_group:label(' '),
    unsafe_charge = lua_group:checkbox('\v\r  Unsafe Recharge In Air'),
    scout_airstop = lua_group:checkbox('\v\r  Scout In Air Auto Stop'),
    scout_airstop_bind = lua_group:hotkey('\v\r  Scout In Air Auto Stop', true),

    ai_peek = fakelag_group:checkbox('\v\r  AI PEEK'),
    dot_offset = fakelag_group:slider('\v\r  Start Offset', 0, 20, 8, true, 'u' ),
    dot_span = fakelag_group:slider('\v\r  Covered offset', 0, 60, 5, true, 'u' ),
    dot_amount = fakelag_group:slider('\v\r  Number of points', 0, 8, 3, true, 'u' ),
}

lua_items.misc = {
    buy_bot = fakelag_group:checkbox('\v\r  Buy Bot'),
    buybot_primary = fakelag_group:combobox('\v\r  Primary Weapon', {"Auto", "Scout", "Awp", "Primary rifle", "Scoped rifle", "Machine gun"}),
    buybot_secondary = fakelag_group:combobox('\v\r  Secondary Weapon', {"Default pistol", "P250", "Dual Berettas", "Light pistol", "Heavy pistol"}),
    buybot_utility = fakelag_group:multiselect('\v\r  Utility', {"Grenade", "Smoke", "Incendiary", "Flashbang", "Kevlar + Helmet", "Defuse kit", "Zeus"}),
    clantag = fakelag_group:checkbox('\v\r  Clantag'),
    fps_boost = fakelag_group:checkbox('\v\r  Fps Boost'),
    fps_always = fakelag_group:checkbox('\v\r  Always On'),
    fps_detect = fakelag_group:multiselect('\v\r  Detections', {'On Peek', 'Hittable'}),
    fps_opt = fakelag_group:multiselect('\v\r  Select', {'3D Sky', 'Fog', 'Shadows', 'Blood', 'Decals', 'Bloom', 'Ragdols', 'Eye Candy', 'Molotov', 'Other'}),
    trashtalk = fakelag_group:checkbox('\v\r  trashtalk(kill/dead)'),
    filter_console = fakelag_group:checkbox('\v\r  Filter Console'),
    enemy_chat = fakelag_group:checkbox('\v\r  Enemy Chat'),
    cs2_view = fakelag_group:checkbox("\v\r  Viewmodel In Scope"),

    airlag = lua_group:checkbox('\v\r  Air Lag'),
    auto_tp = lua_group:checkbox('\v\r  Automatic Lc'),
    auto_tp_key = lua_group:hotkey('\v\r  Automatic Lc', true),
    label1 = lua_group:label(" "),
    fast_ladder = lua_group:checkbox('\v\r  Fast Ladder'),
    duck_on_fd = lua_group:checkbox('\v\r  Allow Duck On FakeDuck'),
    
    label2 = lua_group:label(" "),
    ragebot_logs = lua_group:checkbox('\v\r  Ragebot Logs'),
    ragebot_logs_select = lua_group:multiselect('\v\r  Select\n\n', {'Console', 'Screen'}),
    ragebot_hit = lua_group:color_picker('Ragebot Logs', 186, 196, 249),
    ragebot_miss = lua_group:color_picker('Ragebot Logs Miss', 186, 90, 90),

    label3 = lua_group:label(" "),
    animated_zoom = lua_group:checkbox("\v\r  Animated Zoom"),
    animated_step = lua_group:slider('\v\r  Step', 1, 10, 3, true),
    aspectratio = lua_group:checkbox('\v\r  Aspect Ratio'),
    aspectratio_value = lua_group:slider('\v\r  Value', 0, 200, 0, true, '', 0.01),
    thirdperson = lua_group:checkbox('\v\r  Third Person Distance'),
    thirdperson_value = lua_group:slider('\v\r  Value\n', 30, 150, 150, true, '', 1),

    label4 = lua_group:label(" "),
    anims = lua_group:checkbox("\v\r  Local Animations"),
    anims_ground = lua_group:combobox("\v\r  On Ground", {'Off', 'Static', 'Jitter', 'Jitter Slide', 'MoonWalk'}),
    anims_ground_amount = lua_group:slider('  \v\r  On Ground Amount', 0, 100, 100),
    anims_air = lua_group:combobox("\v\r  In Air", {'Off', 'Static', 'Jitter', 'Better Jitter', 'MoonWalk'}),
    anims_air_amount = lua_group:slider('  \v\r  In Air Amount', 0, 100, 100),
    move_lean = lua_group:combobox("\v\r  Move Lean", {'Off', 'Static', 'Jitter'}),
    move_lean_amount = lua_group:slider('  \v\r  Move Lean Amount', 0, 100, 100),
}

lua_items.visuals = {
    debug_panel = lua_group:checkbox('\v\r  Debug Panel', {208, 185, 245, 200}),
    debug_panel_style = lua_group:combobox('\v\r  Style', {'Default', 'Modern'}),
    label1 = lua_group:label(' '),
    custom_scope = lua_group:checkbox('\v\r  Custom Scope', {200, 200, 200, 200}),
    custom_scope_length = lua_group:slider('\v\r  Length', 50, 500, 200),
    custom_scope_offset = lua_group:slider('\v\r  Offset', 0, 50, 200),
    label3 = lua_group:label(' '),
    manual_ind = lua_group:checkbox('\v\r  Manual Arrows', {200, 200, 200}),
    manual_active = lua_group:checkbox('\v\r  When Active Only', {255, 255, 255, 255}),
    manual_ind_type = lua_group:combobox('\v\r  Type ', {'TS', 'Alternative'}),
    label2 = lua_group:label(' '),
    damage_ind = lua_group:checkbox('\v\r  Damage Indicator', {200, 200, 200, 200}),
    damage_over = lua_group:checkbox('\v\r  If Override'),
    damage_type = lua_group:combobox('\v\r  Font', {'Default', 'Alternative'}),

    bullet_tracers = fakelag_group:checkbox(' \v\r   Bullet Tracers', {150, 150, 150, 150}),
    kibit_hitmarker = fakelag_group:checkbox('\v\r  Kibit Hitmarker', {127, 255, 212}),
    velocity_warning = fakelag_group:checkbox('\v\r  Velocity Warning', {200, 200, 200}),
    defensive_warning = fakelag_group:checkbox('\v\r  Defensive Warning', {200, 200, 200}),
}

lua_items.config = {
    is_show = lua_group:checkbox('###Show'),
    is_show2 = lua_group:checkbox('####Show'),
    list = lua_group:listbox("\vConfigs", ""),
    create = lua_group:button("\v\r  Create", function() end),
    name = lua_group:textbox("\vConfig Name"),
    load = lua_group:button("\v\r    Load", function() end),
    label1 = lua_group:label(' '),
    save = lua_group:button("\v\r    Save", function() 
        lua_items.config.is_show:set(false)
    end),
    accept = lua_group:button("\v\r  \vConfirm", function() 
        lua_items.config.is_show:set(true)
    end),
    decline = lua_group:button("\v\r  \vDecline", function() 
        lua_items.config.is_show:set(true)
    end),
    label2 = lua_group:label(' '),
    label3 = lua_group:label(' '),
    delete = lua_group:button("\v\r  Delete", function() 
        lua_items.config.is_show2:set(false)
    end),
    accept2 = lua_group:button("\v\r  \vConfirm", function() 
        lua_items.config.is_show2:set(true)
    end),
    decline2 = lua_group:button("\v\r  \vDecline", function() 
        lua_items.config.is_show2:set(true)
    end),
    label4 = lua_group:label(' '),
    import = lua_group:button("\v\r  Import", function() end),
    export = lua_group:button("\v\r  Export", function() end),



    online_cfg = fakelag_group:listbox("\vOnline Configs", ""),
    load_cfg = fakelag_group:button('\v\r  Load Config', function() end),
    --upload_cfg = fakelag_group:button('\v\r  Upload My Config', function() end),
    like_button = fakelag_group:button('\v\r  Like', function() end),
    dislike_button = fakelag_group:button('\v\r  Dislike', function() end),
    label_like = fakelag_group:label('  \v\r  Likes: 0 | \v\r  Dislikes: 0'),
}

lua_items.config.is_show:set(true)
lua_items.config.is_show2:set(true)

local function hide_original_menu(state)
    ui.set_visible(ref.enabled, state)
    ui.set_visible(ref.pitch[1], state)
    ui.set_visible(ref.pitch[2], state)
    ui.set_visible(ref.yawbase, state)
    ui.set_visible(ref.yaw[1], state)
    ui.set_visible(ref.yaw[2], state)
    ui.set_visible(ref.yawjitter[1], state)
    ui.set_visible(ref.roll[1], state)
    ui.set_visible(ref.yawjitter[2], state)
    ui.set_visible(ref.bodyyaw[1], state)
    ui.set_visible(ref.bodyyaw[2], state)
    ui.set_visible(ref.fsbodyyaw, state)
    ui.set_visible(ref.edgeyaw, state)
    ui.set_visible(ref.freestand[1], state)
    ui.set_visible(ref.freestand[2], state)

    ui.set_visible(ref.fakelag[1], state)
    ui.set_visible(ref.fakelag[2], state)
    ui.set_visible(ref.fakelag_amount, state)
    ui.set_visible(ref.fakelag_variance, state)
    ui.set_visible(ref.fakelag_limit, state)
    
    ui.set_visible(ref.slowmotion[1], state or (lua_items.main.lol:get() and lua_items.main.tab:get() == 'Anti-Aim'))
    ui.set_visible(ref.slowmotion[2], state or (lua_items.main.lol:get() and lua_items.main.tab:get() == 'Anti-Aim'))
    ui.set_visible(ref.legmovement, state or (lua_items.main.lol:get() and lua_items.main.tab:get() == 'Anti-Aim'))
    ui.set_visible(ref.os[1], state or (lua_items.main.lol:get() and lua_items.main.tab:get() == 'Anti-Aim'))
    ui.set_visible(ref.os[2], state or (lua_items.main.lol:get() and lua_items.main.tab:get() == 'Anti-Aim'))
    ui.set_visible(ref.flickpeek[1], state or (lua_items.main.lol:get() and lua_items.main.tab:get() == 'Anti-Aim'))
    ui.set_visible(ref.flickpeek[2], state or (lua_items.main.lol:get() and lua_items.main.tab:get() == 'Anti-Aim'))
end

local antiaim_builder = {}
local space = {"\n", "\n\n", "\n\n\n", "\n\n\n\n", "\n\n\n\n\n", "\n\n\n\n\n\n", "\n\n\n\n\n\n\n", "\n\n\n\n\n\n\n\n", '\n\n\n\n\n\n\n\n\n', '\n\n\n\n\n\n\n\n\n\n', '\n\n\n\n\n\n\n\n\n\n\n', '\n\n\n\n\n\n\n\n\n\n\n\n'}

for i = 1, #aa_conditions do
    antiaim_builder[i] = {
        override = lua_group:checkbox('\v⚠\r  Override · \v'..aa_conditions[i]),
        yaw = lua_group:slider('\v\r  Yaw Offset'..space[i], -180, 180, 0, true, '°', 1),
        yaw_add = lua_group:checkbox('\v\r  Add L&R Yaw'..space[i]),
        yaw_left = lua_group:slider('\v\r  Yaw Left'..space[i], -180, 180, 0, true, '°', 1),
        yaw_right = lua_group:slider('\v\r  Yaw Right'..space[i], -180, 180, 0, true, '°', 1),
        yaw_random = lua_group:slider('\v\r  Random'..space[i], 0, 100, 0, true, '%', 1),
        label1 = lua_group:label(' '),
        mod_type = lua_group:slider('\v\r  Modifier'..space[i], 1, 5, 1, true, '', 1, {[1] = 'Off', [2] = 'Offset', [3] = 'Center', [4] = 'Random', [5] = 'Skitter'}),
        mod_offset = lua_group:slider('\n\n\n\n\n\n', -180, 180, 0, true, '°', 1),
        mod_addon = lua_group:slider('\n\n\n\n\n\n\n', 1, 3, 1, true, '', 1, {[1] = 'Default', [2] = 'Randomize', [3] = 'Switch'}),
        mod_random = lua_group:slider('\n\n\n\n\n\n\n\n '..space[i], 0, 100, 0, true, '%', 1),
        label2 = lua_group:label(' '),
        body_yaw = lua_group:checkbox('\v\r  Body Yaw'..space[i]),
        body_yaw_type = lua_group:combobox('\v\r  Type'..space[i], {'Opposite', 'Jitter', 'Static'}),
        body_amount = lua_group:slider('\v\r  Amount'..space[i], -180, 180, 0, true, '°', 1),
        body_delay = lua_group:slider('\v\r  Delay'..space[i], 0, 10, 0, true, 't', 1),
        body_delay_random = lua_group:slider('\v\r  Random  '..space[i], 0, 6, 0, true, 't', 1),
        force_lc = lua_group:checkbox('\v\r  Force \vLC'..space[i]),
        snap_aa = lua_group:checkbox('\v\r  Snap AA'..space[i]),
        label3 = lua_group:label(' '),
        defensive_yaw = lua_group:combobox('\v\r  Defensive Yaw'..space[i], {'Off', 'Static', 'Spin', 'Random', 'Random Static', 'Flick Exploit'}), --'L&R'
        defensive_offset = lua_group:slider('\v\r  Yaw Offset '..space[i], -180, 180, 0, true, '°', 1),
        defensive_left = lua_group:slider('\v\r  Yaw Left '..space[i], -180, 180, 0, true, '°', 1),
        defensive_right = lua_group:slider('\v\r  Yaw Right '..space[i], -180, 180, 0, true, '°', 1),
        defensive_spin = lua_group:slider('\v\r  Spin Offset'..space[i], 0, 360, 360, true, '°', 1),
        defensive_speed = lua_group:slider('\v\r  Spin Speed'..space[i], -50, 50, 10, true, 't', 0.1),
        defensive_pitch = lua_group:combobox('\v\r  Defensive Pitch'..space[i], {'Off', 'Offset', 'L&R', 'Random', 'Random Static'}),
        defensive_pitch_offset = lua_group:slider('\v\r  Pitch Offset'..space[i], -89, 89, 0, true, '°', 1),
        defensive_pitch_min = lua_group:slider('\v\r  Pitch Min'..space[i], -89, 89, 0, true, '°', 1),
        defensive_pitch_max = lua_group:slider('\v\r  Pitch Max'..space[i], -89, 89, 0, true, '°', 1),
        defensive_addons = lua_group:multiselect('\v\r  Addons'..space[i], {'Modifier', 'Body Yaw'}),
        defensive_mod_type = lua_group:combobox('\v\r  Modifier '..space[i], {'Off', 'Offset', 'Center', 'Random', 'Skitter'}),
        defensive_mod_offset = lua_group:slider('\v\r  Modifier Offset '..space[i], -180, 180, 0, true, '°', 1),
        defensive_mod_random = lua_group:slider('\v\r  Random   '..space[i], 0, 100, 0, true, '%', 1),
        defensive_body_yaw_type = lua_group:combobox('\v\r  Body Yaw Type'..space[i], {'Opposite', 'Jitter', 'Static'}),
        defensive_body_amount = lua_group:slider('\v\r  Amount '..space[i], -180, 180, 0, true, '°', 1),
        defensive_body_delay = lua_group:slider('\v\r  Delay '..space[i], 0, 10, 0, true, 't', 1),
        defensive_body_delay_random = lua_group:slider('\v\r  Random    '..space[i], 0, 6, 0, true, 't', 1),
    }   
end

lua_items.main.back:set_callback(function()
    lua_items.main.lol:set(false)
end)

lua_items.main.antiaim:set_callback(function()
    lua_items.main.lol:set(true)
    lua_items.main.tab:set('Anti-Aim')
end)

lua_items.main.ragebot:set_callback(function()
    lua_items.main.lol:set(true)
    lua_items.main.tab:set('Ragebot')
end)

lua_items.main.misc:set_callback(function()
    lua_items.main.lol:set(true)
    lua_items.main.tab:set('Misc')
end)

lua_items.main.visuals:set_callback(function()
    lua_items.main.lol:set(true)
    lua_items.main.tab:set('Visuals')
end)

lua_items.main.configs:set_callback(function()
    lua_items.main.lol:set(true)
    lua_items.main.tab:set('Configs')
end)

lua_items.antiaim.main:set_callback(function()
    lua_items.antiaim.tab:set('Main')
end)

lua_items.antiaim.keybinds:set_callback(function()
    lua_items.antiaim.tab:set('Keybinds')
end)

lua_items.antiaim.fakelag:set_callback(function()
    lua_items.antiaim.tab:set('Fake Lag')
end)

lua_items.antiaim.to_def:set_callback(function()
    lua_items.antiaim.is_def:set(true)
end)

lua_items.antiaim.to_build:set_callback(function()
    lua_items.antiaim.is_def:set(false)
end)

local is_show = {lua_items.main.lol, true}
local not_show = {lua_items.main.lol, false}
local is_antiaim = {lua_items.main.tab, 'Anti-Aim'}
local is_ragebot = {lua_items.main.tab, 'Ragebot'}
local is_misc = {lua_items.main.tab, 'Misc'}
local is_visuals = {lua_items.main.tab, 'Visuals'}
local is_configs = {lua_items.main.tab, 'Configs'}
local is_main = {lua_items.antiaim.tab, 'Main'}
local is_keybinds = {lua_items.antiaim.tab, 'Keybinds'}
local is_fakelag = {lua_items.antiaim.tab, 'Fake Lag'}

lua_items.main.lol:depend({lua_items.main.lol, function() return false end})
lua_items.main.tab:depend({lua_items.main.lol, function() return false end})
lua_items.main.back:depend(is_show)
lua_items.main.antiaim:depend(not_show)
lua_items.main.ragebot:depend(not_show)
lua_items.main.misc:depend(not_show)
lua_items.main.visuals:depend(not_show)
lua_items.main.configs:depend(not_show)

lua_items.main.labelsp:depend(not_show)
lua_items.main.labeluser:depend(not_show)
lua_items.main.labelonline:depend(not_show)
lua_items.main.labeltime:depend(not_show)
lua_items.main.labelmiss:depend(not_show)
lua_items.main.labelenem:depend(not_show)
lua_items.main.labelaa:depend(not_show)
lua_items.main.labelrg:depend(not_show)
lua_items.main.labelvs:depend(not_show)
lua_items.main.labelmc:depend(not_show)
lua_items.main.labelcf:depend(not_show)
lua_items.main.labells:depend(not_show)

lua_items.main.discord:depend(not_show)
lua_items.main.league:depend(not_show)
lua_items.main.yt:depend(not_show)
lua_items.main.label_1:depend(not_show)
lua_items.main.label_2:depend(not_show)
lua_items.main.label_3:depend(not_show)

lua_items.main.label3:depend({lua_items.main.tab, function() return lua_items.main.tab:get() ~= 'Anti-Aim' end})
lua_items.antiaim.label1:depend(is_show, is_antiaim)

lua_items.antiaim.main:depend(is_show, is_antiaim)
lua_items.antiaim.keybinds:depend(is_show, is_antiaim)
lua_items.antiaim.fakelag:depend(is_show, is_antiaim)
lua_items.antiaim.tab:depend({lua_items.main.lol, function() return false end})


lua_items.antiaim.addons:depend(is_show, is_antiaim, is_main)
lua_items.antiaim.aa_override:depend(is_show, is_antiaim, is_main, {lua_items.antiaim.addons, 'Override Anti-Aim'})
lua_items.antiaim.safe_head:depend(is_show, is_antiaim, {lua_items.antiaim.addons, 'Safe Head'}, is_main)
lua_items.antiaim.height_difference:depend(is_show, is_antiaim, {lua_items.antiaim.addons, 'Safe Head'}, {lua_items.antiaim.safe_head, 'Height Difference'}, is_main)
lua_items.antiaim.defensive_triggers:depend(is_show, is_antiaim, is_main)
lua_items.antiaim.defensive_condition:depend(is_show, is_antiaim, {lua_items.antiaim.defensive_triggers, function() return lua_items.antiaim.defensive_triggers:get('Hittable') or lua_items.antiaim.defensive_triggers:get('Reload') or lua_items.antiaim.defensive_triggers:get('Flashed') or lua_items.antiaim.defensive_triggers:get('Damaged') or lua_items.antiaim.defensive_triggers:get('Weapon Swap') end}, is_main)

lua_items.antiaim.fl_custom:depend(is_show, is_antiaim, is_fakelag)
lua_items.antiaim.fl_c_type:depend(is_show, is_antiaim, is_fakelag, {lua_items.antiaim.fl_custom, true})
lua_items.antiaim.fl_c_limit:depend(is_show, is_antiaim, is_fakelag, {lua_items.antiaim.fl_custom, true})
lua_items.antiaim.fl_amount:depend(is_show, is_antiaim, is_fakelag, {lua_items.antiaim.fl_custom, false})
lua_items.antiaim.fl_variance:depend(is_show, is_antiaim, is_fakelag, {lua_items.antiaim.fl_custom, false})
lua_items.antiaim.fl_limit:depend(is_show, is_antiaim, is_fakelag, {lua_items.antiaim.fl_custom, false})
lua_items.antiaim.fl_disablers:depend(is_show, is_antiaim, is_fakelag)


lua_items.antiaim.edgeyaw_on_fd:depend(is_show, is_antiaim, is_keybinds)
lua_items.antiaim.freestanding:depend(is_show, is_antiaim, is_keybinds)
lua_items.antiaim.freestanding_key:depend(is_show, is_antiaim, {lua_items.antiaim.freestanding, true}, is_keybinds)
lua_items.antiaim.fr_options:depend(is_show, is_antiaim, {lua_items.antiaim.freestanding, true}, is_keybinds)
lua_items.antiaim.fr_disablers:depend(is_show, is_antiaim, {lua_items.antiaim.freestanding, true}, {lua_items.antiaim.fr_options, 'Disablers'}, is_keybinds)
lua_items.antiaim.manual_keybinds:depend(is_show, is_antiaim, is_keybinds)
lua_items.antiaim.yaw_options:depend(is_show, is_antiaim, {lua_items.antiaim.manual_keybinds, true}, is_keybinds)
lua_items.antiaim.key_left:depend(is_show, is_antiaim, {lua_items.antiaim.manual_keybinds, true}, is_keybinds)
lua_items.antiaim.key_right:depend(is_show, is_antiaim, {lua_items.antiaim.manual_keybinds, true}, is_keybinds)
lua_items.antiaim.key_forward:depend(is_show, is_antiaim, {lua_items.antiaim.manual_keybinds, true}, is_keybinds)

lua_items.antiaim.is_def:depend({lua_items.main.lol, function() return false end})
lua_items.antiaim.to_def:depend(is_show, is_antiaim, {lua_items.antiaim.is_def, false})
lua_items.antiaim.to_build:depend(is_show, is_antiaim, {lua_items.antiaim.is_def, true})

lua_items.antiaim.pitch:depend(is_show, is_antiaim, {lua_items.antiaim.is_def, false})
lua_items.antiaim.yaw_base:depend(is_show, is_antiaim, {lua_items.antiaim.is_def, false})
lua_items.antiaim.condition:depend(is_show, is_antiaim)
lua_items.antiaim.label2:depend(is_show, is_antiaim)

lua_items.ragebot.label1:depend(is_show, is_ragebot)
lua_items.ragebot.label2:depend(is_show, is_ragebot)
lua_items.ragebot.label3:depend(is_show, is_ragebot)
lua_items.ragebot.predict:depend(is_show, is_ragebot)
lua_items.ragebot.predict_key:depend(is_show, is_ragebot, {lua_items.ragebot.predict, true})
lua_items.ragebot.pr_over_clor:depend(is_show, is_ragebot, {lua_items.ragebot.predict, true})
lua_items.ragebot.hideshot_fix:depend(is_show, is_ragebot)
lua_items.ragebot.defensive_fix:depend(is_show, is_ragebot)
lua_items.ragebot.unsafe_charge:depend(is_show, is_ragebot)
lua_items.ragebot.resolver:depend(is_show, is_ragebot)
lua_items.ragebot.def_resovler:depend(is_show, is_ragebot)
lua_items.ragebot.scout_airstop:depend(is_show, is_ragebot)
lua_items.ragebot.scout_airstop_bind:depend(is_show, is_ragebot, {lua_items.ragebot.scout_airstop, true})

lua_items.ragebot.ai_peek:depend(is_show, is_ragebot)
lua_items.ragebot.dot_offset:depend(is_show, is_ragebot, {lua_items.ragebot.ai_peek, true})
lua_items.ragebot.dot_span:depend(is_show, is_ragebot, {lua_items.ragebot.ai_peek, true})
lua_items.ragebot.dot_amount:depend(is_show, is_ragebot, {lua_items.ragebot.ai_peek, true})

lua_items.misc.label1:depend(is_show, is_misc)
lua_items.misc.label2:depend(is_show, is_misc)
lua_items.misc.label3:depend(is_show, is_misc)
lua_items.misc.label4:depend(is_show, is_misc)

lua_items.misc.fps_boost:depend(is_show, is_misc)
lua_items.misc.fps_always:depend(is_show, is_misc, {lua_items.misc.fps_boost, true})
lua_items.misc.fps_detect:depend(is_show, is_misc, {lua_items.misc.fps_boost, true}, {lua_items.misc.fps_always, false})
lua_items.misc.fps_opt:depend(is_show, is_misc, {lua_items.misc.fps_boost, true})

lua_items.misc.buy_bot:depend(is_show, is_misc)
lua_items.misc.buybot_primary:depend(is_show, is_misc, {lua_items.misc.buy_bot, true})
lua_items.misc.buybot_secondary:depend(is_show, is_misc, {lua_items.misc.buy_bot, true})
lua_items.misc.buybot_utility:depend(is_show, is_misc, {lua_items.misc.buy_bot, true})
lua_items.misc.fast_ladder:depend(is_show, is_misc)

lua_items.misc.ragebot_logs:depend(is_show, is_misc)
lua_items.misc.ragebot_logs_select:depend(is_show, is_misc, {lua_items.misc.ragebot_logs, true})
lua_items.misc.ragebot_hit:depend(is_show, is_misc, {lua_items.misc.ragebot_logs, true}, {lua_items.misc.ragebot_logs_select, function() return lua_items.misc.ragebot_logs_select:get('Console') or lua_items.misc.ragebot_logs_select:get('Screen') end})
lua_items.misc.ragebot_miss:depend(is_show, is_misc, {lua_items.misc.ragebot_logs, true}, {lua_items.misc.ragebot_logs_select, function() return lua_items.misc.ragebot_logs_select:get('Console') or lua_items.misc.ragebot_logs_select:get('Screen') end})

lua_items.misc.clantag:depend(is_show, is_misc)
lua_items.misc.trashtalk:depend(is_show, is_misc)
lua_items.misc.filter_console:depend(is_show, is_misc)
lua_items.misc.animated_zoom:depend(is_show, is_misc)
lua_items.misc.animated_step:depend(is_show, is_misc, {lua_items.misc.animated_zoom, true})
lua_items.misc.cs2_view:depend(is_show, is_misc)
lua_items.misc.enemy_chat:depend(is_show, is_misc)
lua_items.misc.duck_on_fd:depend(is_show, is_misc)
lua_items.misc.airlag:depend(is_show, is_misc)
lua_items.misc.auto_tp:depend(is_show, is_misc)
lua_items.misc.auto_tp_key:depend(is_show, is_misc, {lua_items.misc.auto_tp, true})
lua_items.misc.aspectratio:depend(is_show, is_misc)
lua_items.misc.aspectratio_value:depend(is_show, is_misc, {lua_items.misc.aspectratio, true})
lua_items.misc.thirdperson:depend(is_show, is_misc)
lua_items.misc.thirdperson_value:depend(is_show, is_misc, {lua_items.misc.thirdperson, true})

lua_items.misc.anims:depend(is_show, is_misc)
lua_items.misc.anims_ground:depend(is_show, is_misc, {lua_items.misc.anims, true})
lua_items.misc.anims_ground_amount:depend(is_show, is_misc, {lua_items.misc.anims, true}, {lua_items.misc.anims_ground, function() return lua_items.misc.anims_ground:get() ~= 'Off' end})
lua_items.misc.anims_air:depend(is_show, is_misc, {lua_items.misc.anims, true})
lua_items.misc.anims_air_amount:depend(is_show, is_misc, {lua_items.misc.anims, true}, {lua_items.misc.anims_air, function() return lua_items.misc.anims_air:get() ~= 'Off' end})
lua_items.misc.move_lean:depend(is_show, is_misc, {lua_items.misc.anims, true})
lua_items.misc.move_lean_amount:depend(is_show, is_misc, {lua_items.misc.anims, true}, {lua_items.misc.move_lean, function() return lua_items.misc.move_lean:get() ~= 'Off' end})

lua_items.visuals.label1:depend(is_show, is_visuals)
lua_items.visuals.label2:depend(is_show, is_visuals)
lua_items.visuals.label3:depend(is_show, is_visuals)
lua_items.visuals.debug_panel:depend(is_show, is_visuals)
lua_items.visuals.debug_panel_style:depend(is_show, is_visuals, {lua_items.visuals.debug_panel, true})
lua_items.visuals.custom_scope:depend(is_show, is_visuals)
lua_items.visuals.custom_scope_length:depend(is_show, is_visuals, {lua_items.visuals.custom_scope, true})
lua_items.visuals.custom_scope_offset:depend(is_show, is_visuals, {lua_items.visuals.custom_scope, true})
lua_items.visuals.damage_ind:depend(is_show, is_visuals)
lua_items.visuals.damage_over:depend(is_show, is_visuals, {lua_items.visuals.damage_ind, true})
lua_items.visuals.damage_type:depend(is_show, is_visuals, {lua_items.visuals.damage_ind, true})
lua_items.visuals.manual_ind:depend(is_show, is_visuals)
lua_items.visuals.manual_active:depend(is_show, is_visuals, {lua_items.visuals.manual_ind, true})
lua_items.visuals.manual_ind_type:depend(is_show, is_visuals, {lua_items.visuals.manual_ind, true})
lua_items.visuals.velocity_warning:depend(is_show, is_visuals)
lua_items.visuals.defensive_warning:depend(is_show, is_visuals)
lua_items.visuals.bullet_tracers:depend(is_show, is_visuals)
lua_items.visuals.kibit_hitmarker:depend(is_show, is_visuals)

lua_items.config.is_show:depend(is_show, is_configs, {lua_items.config.is_show, function() return false end})
lua_items.config.is_show2:depend(is_show, is_configs, {lua_items.config.is_show2, function() return false end})
lua_items.config.list:depend(is_show, is_configs)
lua_items.config.name:depend(is_show, is_configs)
lua_items.config.create:depend(is_show, is_configs)
lua_items.config.load:depend(is_show, is_configs, {lua_items.config.is_show, true}, {lua_items.config.is_show2, true})
lua_items.config.save:depend(is_show, is_configs, {lua_items.config.is_show, true}, {lua_items.config.is_show2, true})
lua_items.config.accept:depend(is_show, is_configs, {lua_items.config.is_show, false})
lua_items.config.decline:depend(is_show, is_configs, {lua_items.config.is_show, false})
lua_items.config.label1:depend(is_show, is_configs, {lua_items.config.is_show, false})
lua_items.config.label2:depend(is_show, is_configs, {lua_items.config.is_show, false})
lua_items.config.delete:depend(is_show, is_configs, {lua_items.config.is_show2, true}, {lua_items.config.is_show, true})
lua_items.config.accept2:depend(is_show, is_configs, {lua_items.config.is_show2, false})
lua_items.config.decline2:depend(is_show, is_configs, {lua_items.config.is_show2, false})
lua_items.config.label3:depend(is_show, is_configs, {lua_items.config.is_show2, false})
lua_items.config.label4:depend(is_show, is_configs, {lua_items.config.is_show2, false})
lua_items.config.import:depend(is_show, is_configs, {lua_items.config.is_show, true}, {lua_items.config.is_show2, true})
lua_items.config.export:depend(is_show, is_configs, {lua_items.config.is_show, true}, {lua_items.config.is_show2, true})

lua_items.config.online_cfg:depend(is_show, is_configs)
lua_items.config.load_cfg:depend(is_show, is_configs)
--lua_items.config.upload_cfg:depend(is_show, is_configs)
lua_items.config.like_button:depend(is_show, is_configs)
lua_items.config.dislike_button:depend(is_show, is_configs)
lua_items.config.label_like:depend(is_show, is_configs)



for i = 1, #aa_conditions do
    local cond_check = {lua_items.antiaim.condition, function() return (i ~= 1) end}
    local state = {antiaim_builder[i].override, function() if (i == 1) then return true else return antiaim_builder[i].override:get() end end}
    local tab_cond = {lua_items.antiaim.condition, aa_conditions[i]}
    local is_build = {lua_items.antiaim.is_def, false}
    local is_def = {lua_items.antiaim.is_def, true}
    antiaim_builder[i].override:depend(is_show, is_antiaim, cond_check, tab_cond)
    antiaim_builder[i].yaw_add:depend(is_show, is_antiaim, state, tab_cond, is_build)
    antiaim_builder[i].yaw:depend(is_show, is_antiaim, state, tab_cond, is_build)
    antiaim_builder[i].yaw_left:depend(is_show, is_antiaim, state, tab_cond, {antiaim_builder[i].yaw_add, true}, is_build)
    antiaim_builder[i].yaw_right:depend(is_show, is_antiaim, state, tab_cond, {antiaim_builder[i].yaw_add, true}, is_build)
    antiaim_builder[i].yaw_random:depend(is_show, is_antiaim, state, tab_cond, {antiaim_builder[i].yaw_add, true}, is_build)
    antiaim_builder[i].mod_type:depend(is_show, is_antiaim, state, tab_cond, is_build)
    antiaim_builder[i].mod_offset:depend(is_show, is_antiaim, state, tab_cond, {antiaim_builder[i].mod_type, function() return antiaim_builder[i].mod_type:get() ~= 1 end}, is_build)
    antiaim_builder[i].mod_addon:depend(is_show, is_antiaim, state, tab_cond, {antiaim_builder[i].mod_type, function() return antiaim_builder[i].mod_type:get() ~= 1 end}, is_build)
    antiaim_builder[i].mod_random:depend(is_show, is_antiaim, state, tab_cond, {antiaim_builder[i].mod_type, function() return antiaim_builder[i].mod_type:get() ~= 1 end}, {antiaim_builder[i].mod_addon, function() return antiaim_builder[i].mod_addon:get() ~= 1 end}, is_build)
    antiaim_builder[i].body_yaw:depend(is_show, is_antiaim, state, tab_cond, is_build)
    antiaim_builder[i].body_yaw_type:depend(is_show, is_antiaim, state, tab_cond, {antiaim_builder[i].body_yaw, true}, is_build)
    antiaim_builder[i].body_amount:depend(is_show, is_antiaim, state, tab_cond, {antiaim_builder[i].body_yaw, true}, {antiaim_builder[i].body_yaw_type, 'Static'}, is_build)
    antiaim_builder[i].body_delay:depend(is_show, is_antiaim, state, tab_cond, {antiaim_builder[i].body_yaw, true}, {antiaim_builder[i].body_yaw_type, 'Jitter'}, is_build)
    antiaim_builder[i].body_delay_random:depend(is_show, is_antiaim, state, tab_cond, {antiaim_builder[i].body_yaw, true}, {antiaim_builder[i].body_yaw_type, 'Jitter'}, {antiaim_builder[i].body_delay, function() return antiaim_builder[i].body_delay:get() > 0 end}, is_build)

    antiaim_builder[i].force_lc:depend(is_show, is_antiaim, state, tab_cond, is_def)
    antiaim_builder[i].snap_aa:depend(is_show, is_antiaim, state, tab_cond, is_def)

    antiaim_builder[i].defensive_yaw:depend(is_show, is_antiaim, state, tab_cond, {antiaim_builder[i].snap_aa, true}, is_def)
    antiaim_builder[i].defensive_offset:depend(is_show, is_antiaim, state, tab_cond, {antiaim_builder[i].snap_aa, true}, {antiaim_builder[i].defensive_yaw, function() return antiaim_builder[i].defensive_yaw:get() == 'Static' or antiaim_builder[i].defensive_yaw:get() == 'Flick Exploit'end}, is_def)
    antiaim_builder[i].defensive_left:depend(is_show, is_antiaim, state, tab_cond, {antiaim_builder[i].snap_aa, true}, {antiaim_builder[i].defensive_yaw, function() return antiaim_builder[i].defensive_yaw:get() == 'Random' or antiaim_builder[i].defensive_yaw:get() == 'Random Static' end}, is_def)--antiaim_builder[i].defensive_yaw:get() == 'L&R' or 
    antiaim_builder[i].defensive_right:depend(is_show, is_antiaim, state, tab_cond, {antiaim_builder[i].snap_aa, true}, {antiaim_builder[i].defensive_yaw, function() return antiaim_builder[i].defensive_yaw:get() == 'Random' or antiaim_builder[i].defensive_yaw:get() == 'Random Static' end}, is_def)--antiaim_builder[i].defensive_yaw:get() == 'L&R' or 
    antiaim_builder[i].defensive_spin:depend(is_show, is_antiaim, state, tab_cond, {antiaim_builder[i].snap_aa, true}, {antiaim_builder[i].defensive_yaw, 'Spin'}, is_def)
    antiaim_builder[i].defensive_speed:depend(is_show, is_antiaim, state, tab_cond, {antiaim_builder[i].snap_aa, true}, {antiaim_builder[i].defensive_yaw, 'Spin'}, is_def)
    antiaim_builder[i].defensive_pitch:depend(is_show, is_antiaim, state, tab_cond, {antiaim_builder[i].snap_aa, true}, is_def)
    antiaim_builder[i].defensive_pitch_offset:depend(is_show, is_antiaim, state, tab_cond, {antiaim_builder[i].snap_aa, true}, {antiaim_builder[i].defensive_pitch, 'Offset'}, is_def)
    antiaim_builder[i].defensive_pitch_min:depend(is_show, is_antiaim, state, tab_cond, {antiaim_builder[i].snap_aa, true}, {antiaim_builder[i].defensive_pitch, function() return antiaim_builder[i].defensive_pitch:get() ~= 'Off' and antiaim_builder[i].defensive_pitch:get() ~= 'Offset' end}, is_def)
    antiaim_builder[i].defensive_pitch_max:depend(is_show, is_antiaim, state, tab_cond, {antiaim_builder[i].snap_aa, true}, {antiaim_builder[i].defensive_pitch, function() return antiaim_builder[i].defensive_pitch:get() ~= 'Off' and antiaim_builder[i].defensive_pitch:get() ~= 'Offset' end}, is_def)
    
    antiaim_builder[i].defensive_addons:depend(is_show, is_antiaim, state, tab_cond, {antiaim_builder[i].snap_aa, true}, is_def)
    antiaim_builder[i].defensive_mod_type:depend(is_show, is_antiaim, state, tab_cond, {antiaim_builder[i].snap_aa, true}, {antiaim_builder[i].defensive_addons, 'Modifier'}, is_def)
    antiaim_builder[i].defensive_mod_offset:depend(is_show, is_antiaim, state, tab_cond, {antiaim_builder[i].snap_aa, true}, {antiaim_builder[i].defensive_addons, 'Modifier'}, {antiaim_builder[i].defensive_mod_type, function() return antiaim_builder[i].defensive_mod_type:get() ~= 'Off' end}, is_def)
    antiaim_builder[i].defensive_mod_random:depend(is_show, is_antiaim, state, tab_cond, {antiaim_builder[i].snap_aa, true}, {antiaim_builder[i].defensive_addons, 'Modifier'}, {antiaim_builder[i].defensive_mod_type, function() return antiaim_builder[i].defensive_mod_type:get() ~= 'Off' end}, is_def)
    antiaim_builder[i].defensive_body_yaw_type:depend(is_show, is_antiaim, state, tab_cond, {antiaim_builder[i].snap_aa, true}, {antiaim_builder[i].defensive_addons, 'Body Yaw'}, is_def)
    antiaim_builder[i].defensive_body_amount:depend(is_show, is_antiaim, state, tab_cond, {antiaim_builder[i].snap_aa, true}, {antiaim_builder[i].defensive_addons, 'Body Yaw'}, {antiaim_builder[i].defensive_body_yaw_type, 'Static'}, is_def)
    antiaim_builder[i].defensive_body_delay:depend(is_show, is_antiaim, state, tab_cond, {antiaim_builder[i].snap_aa, true}, {antiaim_builder[i].defensive_addons, 'Body Yaw'}, {antiaim_builder[i].defensive_body_yaw_type, 'Jitter'}, is_def)
    antiaim_builder[i].defensive_body_delay_random:depend(is_show, is_antiaim, state, tab_cond, {antiaim_builder[i].snap_aa, true}, {antiaim_builder[i].defensive_addons, 'Body Yaw'}, {antiaim_builder[i].defensive_body_yaw_type, 'Jitter'}, {antiaim_builder[i].defensive_body_delay, function() return antiaim_builder[i].defensive_body_delay:get() > 0 end}, is_def)
    
    antiaim_builder[i].label1:depend(is_show, is_antiaim, state, tab_cond, is_build)
    antiaim_builder[i].label2:depend(is_show, is_antiaim, state, tab_cond, is_build)
    antiaim_builder[i].label3:depend(is_show, is_antiaim, state, tab_cond, is_def)
end

local function check_charge()
    local lp = entity.get_local_player()
    local m_nTickBase = entity.get_prop(lp, 'm_nTickBase')
    local client_latency = client.latency()
    local shift = math.floor(m_nTickBase - globals.tickcount() - 3 - toticks(client_latency) * .5 + .5 * (client_latency * 10))
    local wanted = -14 + (ui.get(ref.doubletap.fakelag_limit) - 1) + 3
    return shift <= wanted
end

anti_knife_dist = function (x1, y1, z1, x2, y2, z2)
    return math.sqrt((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)
end

local function is_vulnerable()
    for _, v in ipairs(entity.get_players(true)) do
        local flags = (entity.get_esp_data(v)).flags
        if bit.band(flags, bit.lshift(1, 11)) ~= 0 then
            return true
        end
    end
    return false
end

local legit_aa = { } do
    local is_interact_traced = false

    local function should_update(cmd, items)
        local me = entity.get_local_player()

        if me == nil then
            return false
        end

        local weapon = entity.get_player_weapon(me)

        if weapon == nil then
            return false
        end

        local weapon_info = csgo_weapons(weapon)

        if weapon_info == nil then
            return false
        end

        local team = entity.get_prop(me, 'm_iTeamNum')
        local my_origin = vector(entity.get_origin(me))

        local is_weapon_bomb = weapon_info.idx == 49

        local is_defusing = entity.get_prop(me, 'm_bIsDefusing') == 1
        local is_rescuing = entity.get_prop(me, 'm_bIsGrabbingHostage') == 1

        local in_bomb_site = entity.get_prop(me, 'm_bInBombZone') == 1

        if is_defusing or is_rescuing then
            return false
        end

        if in_bomb_site and is_weapon_bomb then
            return false
        end

        if team == 3 and cmd.pitch > 15 then
            local bombs = entity.get_all 'CPlantedC4'

            for i = 1, #bombs do
                local bomb = bombs[i]

                local origin = vector(
                    entity.get_origin(bomb)
                )

                local delta = origin - my_origin
                local distancesqr = delta:lengthsqr()

                if distancesqr < (62 * 62) then
                    return false
                end
            end
        end

        local camera = vector(client.camera_angles())
        local forward = vector():init_from_angles(camera:unpack())

        local eye_pos = vector(client.eye_position())
        local end_pos = eye_pos + forward * 128

        local fraction, entindex = client.trace_line(
            me, eye_pos.x, eye_pos.y, eye_pos.z, end_pos.x, end_pos.y, end_pos.z
        )

        if fraction ~= 1 then
            if entindex == -1 then
                return true
            end

            local classname = entity.get_classname(entindex)

            if classname == 'CWorld' then
                return true
            end

            if classname == 'CFuncBrush' then
                return true
            end

            if classname == 'CCSPlayer' then
                return true
            end

            if classname == 'CHostage' then
                local origin = vector(entity.get_origin(entindex))
                local distance = eye_pos:distsqr(origin)

                if distance < (84 * 84) then
                    return false
                end
            end

            if not is_interact_traced then
                is_interact_traced = true
                return false
            end
        end

        return true
    end

    function legit_aa:update(cmd)
        if cmd.in_use == 0 then
            is_interact_traced = false

            return false
        end


        if not should_update(cmd) then
            return false
        end

        cmd.in_use = 0

        return true
    end
end



local defensive_check = {
    lc_left = 0,
    defensive = false,
    tickbase_max = 0,
    last_cmd = 0
}

function aa_helpers.reset_def()
    defensive_check = {
        lc_left = 0,
        defensive = false,
        tickbase_max = 0,
        last_cmd = 0
    }
end

client.set_event_callback('predict_command', function(arg_140_0)
    local lp = entity.get_local_player()
	if not lp or defensive_check.last_cmd ~= arg_140_0.command_number then
		return
	end

	local tickbase = entity.get_prop(lp, "m_nTickBase") or 0

	if math.abs(tickbase - defensive_check.tickbase_max) > 64 then
		defensive_check.tickbase_max = 0
	end

	if tickbase > defensive_check.tickbase_max then
		defensive_check.tickbase_max = tickbase
	end

	defensive_check.lc_left = math.min(14, math.max(0, defensive_check.tickbase_max - tickbase - 1))
	defensive_check.defensive = defensive_check.lc_left > 0
end)

client.set_event_callback('run_command', function(cmd)
	defensive_check.last_cmd = cmd.command_number
end)

function is_defensive_active(lp)
    if not check_charge() then return false end
    return defensive_check.defensive
end

local id = 1   
function aa_helpers.player_state(cmd)
    local lp = entity.get_local_player()
    if lp == nil then return end

    local vecvelocity = { entity.get_prop(lp, 'm_vecVelocity') }
    local flags = entity.get_prop(lp, 'm_fFlags')
    local velocity = math.sqrt(vecvelocity[1]^2+vecvelocity[2]^2)
    local groundcheck = bit.band(flags, 1) == 1
    local jumpcheck = bit.band(flags, 1) == 0 or cmd.in_jump == 1
    local ducked = entity.get_prop(lp, 'm_flDuckAmount') > 0.7
    local duckcheck = ducked or ui.get(ref.fakeduck)
    local slowwalk_key = ui.get(ref.slow[1]) and ui.get(ref.slow[2])

    if jumpcheck and duckcheck then return "Air+C"
    elseif jumpcheck then return "Air"
    elseif duckcheck and velocity > 10 then return "Duck-Moving"
    elseif duckcheck and velocity < 10 then return "Duck"
    elseif groundcheck and slowwalk_key and velocity > 10 then return "Walking"
    elseif groundcheck and velocity > 5 then return "Moving"
    elseif groundcheck and velocity < 5 then return "Stand"
    else return "Global" end
end

local function normalize_yaw(yaw)
    return (yaw + 180) % 360 - 180
end
 
local function randomize_value(original_value, percent)
    local min_range = original_value - (original_value * percent / 100)
    local max_range = original_value + (original_value * percent / 100)
    return math.random(min_range, max_range)
end

function aa_helpers.custom_spin(value, offset)
    if offset == 0 then
        return 0
    end

    if value >= 0 then
        tick = globals.tickcount() * value
        result = (tick % offset) - offset/2
        return result
    else
        tick = globals.tickcount() * value
        result = (tick % -offset) + offset/2
        return result
    end
end

local yaw_keybinds = 0
local last_press_t_dir = 0
local is_freestand = false
local is_static = false

local function run_keybinds(cmd)
    local lp = entity.get_local_player()
    if lp == nil then return end
    local vecvelocity = { entity.get_prop(lp, 'm_vecVelocity') }
    local flags = entity.get_prop(lp, 'm_fFlags')
    local jumpcheck = bit.band(flags, 1) == 0 or cmd.in_jump == 1
    local moving = math.sqrt(vecvelocity[1]^2+vecvelocity[2]^2) > 10
    local ducked = entity.get_prop(lp, 'm_flDuckAmount') > 0.7
    local duckcheck = ducked or ui.get(ref.fakeduck)

    local is_walking = moving and ui.get(ref.slow[1]) and ui.get(ref.slow[2])
    local is_crouching = duckcheck and not jumpcheck

    local fr_disabler = lua_items.antiaim.fr_options:get('Disablers')

    is_freestand = lua_items.antiaim.freestanding:get() and lua_items.antiaim.freestanding_key:get()
    local is_quick_peek = ui.get(ref.quick_peek[1]) and ui.get(ref.quick_peek[2])

    if yaw_keybinds ~= 0 then
        ui.set(ref.freestand[1], false)
        is_freestand = false
    elseif (fr_disabler and lua_items.antiaim.fr_disablers:get('Walking') and is_walking) or (fr_disabler and lua_items.antiaim.fr_disablers:get('Crouch') and is_crouching) or (fr_disabler and lua_items.antiaim.fr_disablers:get('Air') and jumpcheck) then
        ui.set(ref.freestand[1], false)
        is_freestand = false
    elseif is_quick_peek and lua_items.antiaim.fr_options:get('On Quick Peek') then
        ui.set(ref.freestand[1], true)
        ui.set(ref.freestand[2], 'Always on')
        is_freestand = true
    elseif yaw_keybinds == 0 then
        ui.set(ref.freestand[1], lua_items.antiaim.freestanding:get() and lua_items.antiaim.freestanding_key:get())
        ui.set(ref.freestand[2], lua_items.antiaim.freestanding_key:get() and 'Always on' or 'On hotkey')
        is_freestand = lua_items.antiaim.freestanding:get() and lua_items.antiaim.freestanding_key:get()
    else
        ui.set(ref.freestand[1], false)
        is_freestand = false
    end

    is_static = (lua_items.antiaim.fr_options:get('Disable Yaw Modifier') and is_freestand) or (yaw_keybinds ~= 0 and lua_items.antiaim.yaw_options:get('Disable Yaw Modifier'))

    if lua_items.antiaim.manual_keybinds:get() and lua_items.antiaim.key_right:get() and last_press_t_dir + 0.2 < globals.curtime() then
        yaw_keybinds = yaw_keybinds == 90 and 0 or 90
        last_press_t_dir = globals.curtime()
    elseif lua_items.antiaim.manual_keybinds:get() and lua_items.antiaim.key_left:get() and last_press_t_dir + 0.2 < globals.curtime() then
        yaw_keybinds = yaw_keybinds == -90 and 0 or -90
        last_press_t_dir = globals.curtime()
    elseif lua_items.antiaim.manual_keybinds:get() and lua_items.antiaim.key_forward:get() and last_press_t_dir + 0.2 < globals.curtime() then
        yaw_keybinds = yaw_keybinds == 180 and 0 or 180
        last_press_t_dir = globals.curtime()
    elseif last_press_t_dir > globals.curtime() then
        last_press_t_dir = globals.curtime()
    end

    if not lua_items.antiaim.manual_keybinds:get() then
        yaw_keybinds = 0
    end
end

local current_tickcount = 0
local to_jitter = false
local yaw_amount = 0
local fl_ticks = 0
local last_yaw = 0
local alive_players = {}
local last_def_yaw = 0
local last_def_pitch = 0
local desync_side = false
local mod_value = 0

local choke  = 1
local is_switch = false
local last_pack = 0 

function aa_helpers.static(yaw)
    if yaw_keybinds == 0 then
        ui.set(ref.yaw[2], yaw or 16)
    end
    ui.set(ref.yawjitter[1], 'Off')
    ui.set(ref.yawjitter[2], 0)
    ui.set(ref.bodyyaw[1], 'Static')
    ui.set(ref.bodyyaw[2], 1)
    if id ~= 10 then
        ui.set(ref.pitch[1], "Custom")
        ui.set(ref.pitch[2], 89) 
    end
end

function aa_helpers.setup(cmd)
    local lp = entity.get_local_player()
    if lp == nil then return end
    ui.set(ref.enabled, true)
    if antiaim_builder[10].override:get() and legit_aa:update(cmd) then id = 10
    elseif antiaim_builder[9].override:get() and not check_charge() then id = 9
    elseif antiaim_builder[8].override:get() and aa_helpers.player_state(cmd) == 'Air+C' then id = 8
    elseif antiaim_builder[7].override:get() and aa_helpers.player_state(cmd) == 'Air' then id = 7
    elseif antiaim_builder[6].override:get() and aa_helpers.player_state(cmd) == 'Duck-Moving' then id = 6
    elseif antiaim_builder[5].override:get() and aa_helpers.player_state(cmd) == 'Duck' then id = 5
    elseif antiaim_builder[4].override:get() and aa_helpers.player_state(cmd) == 'Walking' then id = 4
    elseif antiaim_builder[3].override:get() and aa_helpers.player_state(cmd) == 'Moving' then id = 3
    elseif antiaim_builder[2].override:get() and aa_helpers.player_state(cmd) == 'Stand' then id = 2
    else id = 1 end

    ui.set(ref.roll[1], 0)
    ui.set(ref.fsbodyyaw, false)
    ui.set(ref.yaw[1], '180')
    ui.set(ref.pitch[1], lua_items.antiaim.pitch:get())
    ui.set(ref.yawbase, yaw_keybinds == 0 and lua_items.antiaim.yaw_base:get() or 'Local view')
    run_keybinds(cmd)

    local flags = entity.get_prop(lp, 'm_fFlags')
    local is_fd = ui.get(ref.fakeduck)
    local is_osaa = ui.get(ref.os[2]) and ui.get(ref.os[1])
    local is_dt = ui.get(ref.dt[1]) and ui.get(ref.dt[2])
    local ducked = entity.get_prop(lp, 'm_flDuckAmount') > 0.7
    local jumpcheck = bit.band(flags, 1) == 0 or cmd.in_jump == 1
    local vecvelocity = { entity.get_prop(lp, 'm_vecVelocity') }
    local velocity = math.sqrt(vecvelocity[1]^2+vecvelocity[2]^2)
    local groundcheck = bit.band(flags, 1) == 1

    if (lua_items.ragebot.hideshot_fix:get() and is_osaa) and not is_fd and not is_dt then
        ui.set(ref.fakelag[1], false)
        ui.set(ref.fakelag[2], 'On Hotkey')
        ui.set(ref.fakelag_limit, 1)
    else
        if lua_items.antiaim.fl_custom:get() then
            local limit = lua_items.antiaim.fl_c_limit:get()
            
            if lua_items.antiaim.fl_c_type:get() == 'Cycle' then
                limit = math.clamp(limit - fl_ticks % 5, 1, 15)
            else
                limit = math.clamp(limit - math.random(0, 5), 1, 15)
            end

            if is_fd then
                limit = 15
            end

            if not (is_dt and is_osaa and is_fd) then
                if cmd.chokedcommands < limit then
                    cmd.allow_send_packet = false
                else
                    fl_ticks = fl_ticks + 1
                    cmd.no_choke = true
                end
            end

            ui.set(ref.fakelag[1], limit ~= 1)
            ui.set(ref.fakelag[2], 'Always On')
            ui.set(ref.fakelag_amount, 'Maximum')
            ui.set(ref.fakelag_variance, 100)
            ui.set(ref.fakelag_limit, limit)

        else
            ui.set(ref.fakelag[1], true)
            ui.set(ref.fakelag[2], 'Always On')
            ui.set(ref.fakelag_amount, lua_items.antiaim.fl_amount:get())
            ui.set(ref.fakelag_variance, lua_items.antiaim.fl_variance:get())
            ui.set(ref.fakelag_limit, lua_items.antiaim.fl_limit:get())
        end
    end

    if lua_items.antiaim.fl_disablers:get('Stand') and (groundcheck and velocity < 10) then
        if is_osaa or is_dt or ducked or is_fd then
            --smth
        else
            ui.set(ref.fakelag_limit, 1)
        end
    elseif lua_items.antiaim.fl_disablers:get('Crouch') and (groundcheck and ducked) then
        if is_fd or is_osaa or is_dt then
            --smth
        else
            ui.set(ref.fakelag_limit, 1)
        end
    end

    if globals.chokedcommands() == 0 then
        choke = choke + 1
    end

    if (choke - last_pack >= 1)  then
        is_switch = not is_switch
        last_pack = choke
    end

    if antiaim_builder[id].snap_aa:get() and antiaim_builder[id].defensive_yaw:get() == 'Flick Exploit' then
        cmd.force_defensive = cmd.command_number % 7 == 0
    else
        cmd.force_defensive = antiaim_builder[id].force_lc:get()
    end

    local desync_type = entity.get_prop(lp, 'm_flPoseParameter', 11) * 120 - 60
    local desync_side = desync_type > 0

    if antiaim_builder[id].yaw_add:get() then
        yaw_amount = antiaim_builder[id].yaw:get() + (desync_side and randomize_value(antiaim_builder[id].yaw_left:get(), antiaim_builder[id].yaw_random:get()) or randomize_value(antiaim_builder[id].yaw_right:get(), antiaim_builder[id].yaw_random:get()))
    else
        yaw_amount = antiaim_builder[id].yaw:get()
    end

    if antiaim_builder[id].mod_type:get() == 1 then
        ui.set(ref.yawjitter[1], 'Off')
    elseif antiaim_builder[id].mod_type:get() == 2 then
        ui.set(ref.yawjitter[1], 'Offset')
    elseif antiaim_builder[id].mod_type:get() == 3 then
        ui.set(ref.yawjitter[1], 'Center')
    elseif antiaim_builder[id].mod_type:get() == 4 then
        ui.set(ref.yawjitter[1], 'Random')
    elseif antiaim_builder[id].mod_type:get() == 5 then
        ui.set(ref.yawjitter[1], 'Skitter')
    end

    if antiaim_builder[id].mod_addon:get() == 1 then
        mod_value = antiaim_builder[id].mod_offset:get()
    elseif antiaim_builder[id].mod_addon:get() == 2 then
        mod_value = randomize_value(antiaim_builder[id].mod_offset:get(), antiaim_builder[id].mod_random:get())
    elseif antiaim_builder[id].mod_addon:get() == 3 then
        mod_value = globals.tickcount() % math.random(3, 6) > 2 and randomize_value(antiaim_builder[id].mod_offset:get(), antiaim_builder[id].mod_random:get()) or antiaim_builder[id].mod_offset:get()
    end


    ui.set(ref.yawjitter[2], math.clamp(mod_value, -180, 180))

    if antiaim_builder[id].body_yaw:get() then
        if antiaim_builder[id].body_yaw_type:get() == 'Jitter' and antiaim_builder[id].body_delay:get() > 0 then
           
            if globals.tickcount() > current_tickcount + antiaim_builder[id].body_delay:get() + math.random(0, antiaim_builder[id].body_delay_random:get()) + 2 then
                if cmd.chokedcommands == 0 then
                    to_jitter = not to_jitter
                    current_tickcount = globals.tickcount()
                end
            elseif globals.tickcount() <  current_tickcount then
                current_tickcount = globals.tickcount()
            end
            
            ui.set(ref.bodyyaw[1], "Static")
            ui.set(ref.bodyyaw[2], to_jitter and 1 or -1)
            ui.set(ref.yawjitter[1], 'Off')
            ui.set(ref.yawjitter[2], 0)

            local yaw_l = antiaim_builder[id].yaw_add:get() and antiaim_builder[id].yaw:get() + randomize_value(antiaim_builder[id].yaw_right:get(), antiaim_builder[id].yaw_random:get()) or antiaim_builder[id].yaw:get()
            local yaw_r = antiaim_builder[id].yaw_add:get() and antiaim_builder[id].yaw:get() + randomize_value(antiaim_builder[id].yaw_left:get(), antiaim_builder[id].yaw_random:get()) or antiaim_builder[id].yaw:get()
            local jit_amount = mod_value

            if antiaim_builder[id].mod_type:get() == 3 then
                yaw_amount = to_jitter and (yaw_l + jit_amount/2) or (yaw_r - jit_amount/2)
            elseif antiaim_builder[id].mod_type:get() == 2 then
                yaw_amount = to_jitter and (yaw_l) or (yaw_r - jit_amount/2)
            elseif antiaim_builder[id].mod_type:get() == 4 then
                yaw_amount = to_jitter and (yaw_l + jit_amount/2) or (yaw_r - jit_amount/2)
            elseif antiaim_builder[id].mod_type:get() == 5 then
                if globals.tickcount() % 3 == 0 then 
                    yaw_amount = to_jitter and (yaw_l + jit_amount/2) or (yaw_r)
                elseif globals.tickcount() % 3 == 1 then 
                    yaw_amount = to_jitter and (yaw_l) or (yaw_r)
                else
                    yaw_amount = to_jitter and (yaw_l) or (yaw_r - jit_amount/2)
                end
            else
                yaw_amount = to_jitter and yaw_l or yaw_r
            end
        else
            ui.set(ref.bodyyaw[1], antiaim_builder[id].body_yaw_type:get())
            ui.set(ref.bodyyaw[2], antiaim_builder[id].body_yaw_type:get() == 'Jitter' and 1 or antiaim_builder[id].body_amount:get())
        end
    else
        ui.set(ref.bodyyaw[1], 'Off')
    end

    if not is_defensive_active() then
        last_def_yaw = math.random(antiaim_builder[id].defensive_left:get(), antiaim_builder[id].defensive_right:get())
        last_def_pitch = math.random(antiaim_builder[id].defensive_pitch_min:get(), antiaim_builder[id].defensive_pitch_max:get())
    end

    if antiaim_builder[id].snap_aa:get() and is_defensive_active() then
        
        if antiaim_builder[id].defensive_addons:get('Modifier') then
            if antiaim_builder[id].defensive_body_yaw_type:get() == 'Jitter' and antiaim_builder[id].defensive_body_delay:get() > 1 then
                ui.set(ref.yawjitter[1], 'Off')
                ui.set(ref.yawjitter[2], 0)
                local jit_amount = randomize_value(antiaim_builder[id].defensive_mod_offset:get(), antiaim_builder[id].defensive_mod_random:get())
    
                if antiaim_builder[id].defensive_mod_type:get() == 'Center' then
                    yaw_amount = to_jitter and (yaw_amount + jit_amount/2) or (yaw_amount - jit_amount/2)
                elseif antiaim_builder[id].defensive_mod_type:get() == 'Offset' then
                    yaw_amount = to_jitter and (yaw_amount) or (yaw_amount + jit_amount/2)
                elseif antiaim_builder[id].defensive_mod_type:get() == 'Random' then
                    yaw_amount = to_jitter and (yaw_amount + jit_amount/2) or (yaw_amount - jit_amount/2)
                elseif antiaim_builder[id].defensive_mod_type:get() == 'Skitter' then
                    if globals.tickcount() % 3 == 0 then 
                        yaw_amount = to_jitter and (yaw_amount + jit_amount/2) or (yaw_amount)
                    elseif globals.tickcount() % 3 == 1 then 
                        yaw_amount = yaw_amount
                    else
                        yaw_amount = to_jitter and (yaw_amount) or (yaw_amount - jit_amount/2)
                    end
                else
                    yaw_amount = 0
                end
            else
                ui.set(ref.yawjitter[1], antiaim_builder[id].defensive_mod_type:get())
                ui.set(ref.yawjitter[2], math.clamp(randomize_value(antiaim_builder[id].defensive_mod_offset:get(), antiaim_builder[id].defensive_mod_random:get()), -180, 180))
            end
        end

        if antiaim_builder[id].defensive_addons:get('Body Yaw') then
            if antiaim_builder[id].defensive_body_yaw_type:get() == 'Jitter' and antiaim_builder[id].defensive_body_delay:get() > 1 then
                if globals.tickcount() > current_tickcount + antiaim_builder[id].defensive_body_delay:get() + math.random(0, antiaim_builder[id].defensive_body_delay_random:get()) + 2 then
                    if cmd.chokedcommands == 0 then
                        to_jitter = not to_jitter
                        current_tickcount = globals.tickcount()
                    end
                elseif globals.tickcount() <  current_tickcount then
                    current_tickcount = globals.tickcount()
                end

                ui.set(ref.bodyyaw[1], "Static")
                ui.set(ref.bodyyaw[2], to_jitter and 1 or -1)
            else
                ui.set(ref.bodyyaw[1], antiaim_builder[id].defensive_body_yaw_type:get())
                ui.set(ref.bodyyaw[2], antiaim_builder[id].defensive_body_yaw_type:get() == 'Jitter' and 1 or antiaim_builder[id].defensive_body_amount:get())
            end
        end 

        if antiaim_builder[id].defensive_yaw:get() == 'Static' then
            yaw_amount = antiaim_builder[id].defensive_offset:get()
        --elseif antiaim_builder[id].defensive_yaw:get() == 'L&R' then
            --yaw_amount = is_switch and antiaim_builder[id].defensive_left:get() or antiaim_builder[id].defensive_right:get()
        elseif antiaim_builder[id].defensive_yaw:get() == 'Spin' then
            yaw_amount = aa_helpers.custom_spin(antiaim_builder[id].defensive_speed:get(), antiaim_builder[id].defensive_spin:get())
        elseif antiaim_builder[id].defensive_yaw:get() == 'Random' then    
            yaw_amount = math.random(antiaim_builder[id].defensive_left:get(), antiaim_builder[id].defensive_right:get())
        elseif antiaim_builder[id].defensive_yaw:get() == 'Flick Exploit' then
            yaw_amount = antiaim_builder[id].defensive_offset:get()
        elseif antiaim_builder[id].defensive_yaw:get() == 'Random Static' then
            yaw_amount = last_def_yaw
        end

        if antiaim_builder[id].defensive_pitch:get() == 'Offset' then
            ui.set(ref.pitch[1], 'Custom')
            ui.set(ref.pitch[2], antiaim_builder[id].defensive_pitch_offset:get())
        elseif antiaim_builder[id].defensive_pitch:get() == 'L&R' then
            ui.set(ref.pitch[1], 'Custom')
            ui.set(ref.pitch[2], desync_side and antiaim_builder[id].defensive_pitch_min:get() or antiaim_builder[id].defensive_pitch_max:get())
        elseif antiaim_builder[id].defensive_pitch:get() == 'Random' then
            ui.set(ref.pitch[1], 'Custom')
            ui.set(ref.pitch[2], math.random(antiaim_builder[id].defensive_pitch_min:get(), antiaim_builder[id].defensive_pitch_max:get()))
        elseif antiaim_builder[id].defensive_pitch:get() == 'Random Static' then
            ui.set(ref.pitch[1], 'Custom')
            ui.set(ref.pitch[2], last_def_pitch)
        end
    end

    ui.set(ref.yaw[2], yaw_keybinds == 0 and math.clamp(yaw_amount, -180, 180) or yaw_keybinds)

    for i=1, 64 do
        if entity.is_alive(i) and entity.is_enemy(i) then
            table.insert(alive_players, i)
        end
    end

    if lua_items.antiaim.addons:get('Override Anti-Aim') then
        if lua_items.antiaim.aa_override:get('No Enemies Alive') then
            if client.current_threat() == nil and #alive_players == 0 then
                ui.set(ref.yaw[2], globals.tickcount() % 36 * 10 - 180)
                ui.set(ref.yawjitter[2], 0)
                ui.set(ref.bodyyaw[1], 'Static')
                ui.set(ref.bodyyaw[2], 0)
                ui.set(ref.pitch[1], "Custom")
                ui.set(ref.pitch[2], 0) 
                cmd.force_defensive = false
            end
        end

        if lua_items.antiaim.aa_override:get('Warmup') then
            if entity.get_prop(entity.get_game_rules(), "m_bWarmupPeriod") == 1 then
                ui.set(ref.yaw[2], globals.tickcount() % 36 * 10 - 180)
                ui.set(ref.yawjitter[2], 0)
                ui.set(ref.bodyyaw[1], 'Static')
                ui.set(ref.bodyyaw[2], 0)
                ui.set(ref.pitch[1], "Custom")
                ui.set(ref.pitch[2], 0) 
                cmd.force_defensive = false
            end
        end
    end

    alive_players = {}

    local lp_weapon = entity.get_player_weapon(lp)
    local lp_orig_x, lp_orig_y, lp_orig_z = entity.get_prop(lp, "m_vecOrigin")
    local threat = client.current_threat()

    if lua_items.antiaim.addons:get('Safe Head') then
        if lp_weapon ~= nil then
            if lua_items.antiaim.safe_head:get("Air+C Knife") then
                if jumpcheck and ducked and entity.get_classname(lp_weapon) == "CKnife" then
                    aa_helpers.static()
                end
            end
            if lua_items.antiaim.safe_head:get("Air+C Zeus") then
                if jumpcheck and ducked and entity.get_classname(lp_weapon) == "CWeaponTaser" then
                    aa_helpers.static()
                end
            end
            if lua_items.antiaim.safe_head:get("Height Difference") then
                if threat ~= nil and is_vulnerable() then
                    threat_x, threat_y, threat_z = entity.get_prop(threat, "m_vecOrigin")
                    threat_dist = lp_orig_z - threat_z
                    if threat_dist > lua_items.antiaim.height_difference:get() then
                        aa_helpers.static()
                    end
                end
            end
            
            if lua_items.antiaim.safe_head:get("Air+C SMG") then
                if jumpcheck and ducked and (entity.get_classname(lp_weapon) == "CWeaponMAC10" or entity.get_classname(lp_weapon) == "CWeaponMP9" or entity.get_classname(lp_weapon) == "CWeaponMP7" or entity.get_classname(lp_weapon) == "CWeaponUMP45" or entity.get_classname(lp_weapon) == "CWeaponBizon" or entity.get_classname(lp_weapon) == "CWeaponP90") then
                    aa_helpers.static()
                end
            end
        end
    end

    local players = entity.get_players(true)

    if lua_items.antiaim.addons:get('Anti Backstab') then
        for i=1, #players do
            if players == nil then return end
            enemy_orig_x, enemy_orig_y, enemy_orig_z = entity.get_prop(players[i], "m_vecOrigin")
            distance_to = anti_knife_dist(lp_orig_x, lp_orig_y, lp_orig_z, enemy_orig_x, enemy_orig_y, enemy_orig_z)
            weapon = entity.get_player_weapon(players[i])
            if weapon == nil then return end
            if entity.get_classname(weapon) == "CKnife" and distance_to <= 220 then
                ui.set(ref.yaw[2], 180)
                ui.set(ref.yawbase, "At targets")
            end
        end
    end

    if is_freestand and lua_items.antiaim.fr_options:get('Flick peek') and yaw_keybinds == 0 then
        if not cmd.quick_stop then
            if is_vulnerable() then
                cmd.force_defensive = true
                if not is_defensive_active(lp) then
                    last_yaw = entity.get_prop(lp, 'm_flLowerBodyYawTarget')
                else    
                    cmd.pitch = (math.random(-10, 10))
                    if last_yaw > 0 then
                        cmd.yaw = normalize_yaw(last_yaw - 180)
                    else
                        cmd.yaw = normalize_yaw(last_yaw + 180)
                    end
                end
            end
        end
    end

    if lua_items.antiaim.yaw_options:get('Flick peek') and yaw_keybinds ~= 0 then
        if not cmd.quick_stop then
            cmd.force_defensive = true
            if is_defensive_active(lp) then
                ui.set(ref.yaw[1], '180')
                ui.set(ref.yaw[2], -yaw_keybinds)
                ui.set(ref.pitch[2], math.random(-10, 10))
            end
        end
    end

    if lua_items.antiaim.yaw_options:get('Disable Yaw Modifier') and yaw_keybinds ~= 0 then
        aa_helpers.static()
    end

    if is_static and lua_items.antiaim.fr_options:get('Disable Yaw Modifier') and yaw_keybinds == 0 then
        aa_helpers.static()
    end

    if id == 10 then
        ui.set(ref.fsbodyyaw, true)
        ui.set(ref.pitch[1], 'Off')
        ui.set(ref.yaw[1], '180')
        ui.set(ref.yaw[2], normalize_yaw(180 + yaw_amount))
        ui.set(ref.yawbase, 'Local view')
    end

    local animlayers = ffi_helpers.animlayers:get(lp)
    if not animlayers then return end


    local weapon_activity_number = ffi_helpers.activity:get(animlayers[1]['sequence'], me)
    local flash_activity_number = ffi_helpers.activity:get(animlayers[9]['sequence'], me)
    local is_reloading = animlayers[1]['weight'] ~= 0.0 and weapon_activity_number == 967
    local is_flashed = animlayers[9]['weight'] > 0.1 and flash_activity_number == 960
    local is_under_attack = animlayers[10]['weight'] > 0.1
    local is_swapping_weapons = cmd.weaponselect > 0

    local is_stand = lua_items.antiaim.defensive_condition:get('Stand') and aa_helpers.player_state(cmd) == 'Stand'
    local is_moving = lua_items.antiaim.defensive_condition:get('Stand') and aa_helpers.player_state(cmd) == 'Moving'
    local is_walking = lua_items.antiaim.defensive_condition:get('Walking') and aa_helpers.player_state(cmd) == 'Walking'
    local is_crouching = lua_items.antiaim.defensive_condition:get('Crouching') and (aa_helpers.player_state(cmd) == 'Duck' or aa_helpers.player_state(cmd) == 'Duck-Moving')
    local is_air = lua_items.antiaim.defensive_condition:get('Air') and (aa_helpers.player_state(cmd) == 'Air' or aa_helpers.player_state(cmd) == 'Air+C')

    if is_stand or is_moving or is_walking or is_crouching or is_air then
        if lua_items.antiaim.defensive_triggers:get('Hittable') and is_vulnerable() then
            cmd.force_defensive = true
        end
        
        if lua_items.antiaim.defensive_triggers:get('Reload') and is_reloading then
            cmd.force_defensive = true
        end

        if lua_items.antiaim.defensive_triggers:get('Flashed') and is_flashed then
            cmd.force_defensive = true
        end

        if lua_items.antiaim.defensive_triggers:get('Damaged') and is_under_attack then
            cmd.force_defensive = true
        end

        if lua_items.antiaim.defensive_triggers:get('Weapon Swap') and is_swapping_weapons then
            cmd.force_defensive = true
        end
    end
end

local lastmiss = 0
local function GetClosestPoint(A, B, P)
    a_to_p = { P[1] - A[1], P[2] - A[2] }
    a_to_b = { B[1] - A[1], B[2] - A[2] }

    atb2 = a_to_b[1]^2 + a_to_b[2]^2

    atp_dot_atb = a_to_p[1]*a_to_b[1] + a_to_p[2]*a_to_b[2]
    t = atp_dot_atb / atb2
    
    return { A[1] + a_to_b[1]*t, A[2] + a_to_b[2]*t }
end
            
client.set_event_callback("bullet_impact", function(e)  
    if not entity.is_alive(entity.get_local_player()) then return end
    local ent = client.userid_to_entindex(e.userid)
    if ent ~= client.current_threat() then return end
    if entity.is_dormant(ent) or not entity.is_enemy(ent) then return end

    local ent_origin = { entity.get_prop(ent, "m_vecOrigin") }
    ent_origin[3] = ent_origin[3] + entity.get_prop(ent, "m_vecViewOffset[2]")
    local local_head = { entity.hitbox_position(entity.get_local_player(), 0) }
    local closest = GetClosestPoint(ent_origin, { e.x, e.y, e.z }, local_head)
    local delta = { local_head[1]-closest[1], local_head[2]-closest[2] }
    local delta_2d = math.sqrt(delta[1]^2+delta[2]^2)
    if math.abs(delta_2d) <= 60 and globals.curtime() - lastmiss > 0.015 then
        user_info[2] = user_info[2] + 1
        lua_items.main.labelmiss:set(' \v\r  Missed Shots At You: \v'..user_info[2])
        lastmiss = globals.curtime()
    end
end)

local rgba_to_hex = function(b, c, d, e)
    return string.format('%02x%02x%02x%02x', b, c, d, e)
end


local function text_fade_animation(x, y, speed, color1, color2, text, flag)
    local final_text = ''
    local curtime = globals.curtime()
    for i = 0, #text do
        local x = i * 10  
        local wave = math.cos(8 * speed * curtime + x / 30)
        local color = rgba_to_hex(
            math.lerp(color1.r, color2.r, math.clamp(wave, 0, 1)),
            math.lerp(color1.g, color2.g, math.clamp(wave, 0, 1)),
            math.lerp(color1.b, color2.b, math.clamp(wave, 0, 1)),
            color1.a
        ) 
        final_text = final_text .. '\a' .. color .. text:sub(i, i) 
    end
    
    renderer.text(x, y, color1.r, color1.g, color1.b, color1.a, flag, nil, final_text)
end

local function text_fade_menu(speed, color1, color2, text)
    local final_text = ''
    local curtime = globals.curtime()
    for i = 0, #text do
        local x = i * #text  
        local wave = math.cos(8 * speed * curtime + x / 30)
        local color = rgba_to_hex(
            math.lerp(color1.r, color2.r, math.clamp(wave, 0, 1)),
            math.lerp(color1.g, color2.g, math.clamp(wave, 0, 1)),
            math.lerp(color1.b, color2.b, math.clamp(wave, 0, 1)),
            color1.a
        ) 
        final_text = final_text .. '\a' .. color .. text:sub(i, i) 
    end
    return final_text
end

local screen = {client.screen_size()}
local center = {screen[1]/2, screen[2]/2} 

local function mouse_in_bounds(x, y, width, height)
    local mouse_x, mouse_y = ui.mouse_position()
    return mouse_x >= x and mouse_x <= x + width and mouse_y >= y and mouse_y <= y + height
end

local drag_dmg = dragging.new("drag_damage", screen[1] / 2 + 5, screen[2] / 2 - 15)
local dmg_alpha = 0
local dmg_set = 0

function visuals_helpers.damage_override()
    local r,g,b,a = lua_items.visuals.damage_ind:get_color()
    local dmg_ovr = (ui.get(ref.minimum_damage_override[1]) and ui.get(ref.minimum_damage_override[2]))
    local x, y = drag_dmg:get()
    if lua_items.visuals.damage_over:get() then
        if not dmg_ovr then return end
        renderer.text(x, y, r,g,b,a, lua_items.visuals.damage_type:get() == 'Default' and "d" or '-d', 0, ui.get(ref.minimum_damage_override[3]) .. "")
    else
        if dmg_ovr then
            renderer.text(x+2, y+2, r,g,b,a, lua_items.visuals.damage_type:get() == 'Default' and "d" or '-d', 0, ui.get(ref.minimum_damage_override[3]) .. "")
        else
            renderer.text(x+2, y+2, r,g,b,a, lua_items.visuals.damage_type:get() == 'Default' and "d" or '-d', 0, ui.get(ref.minimum_damage) .. "")
        end
    end

    dmg_alpha = math.lerp(dmg_alpha, ui.is_menu_open() and 155 or 0, 0.05)
    dmg_set = math.lerp(dmg_set, (mouse_in_bounds(x, y, 25, 15) and ui.is_menu_open()) and 155 or 0, 0.05)

    if mouse_in_bounds(x, y, 25, 15) and client.key_state(0x02) == true then
        drag_dmg:set(screen[1] / 2 + 5, screen[2] / 2 - 15)
    end

    if dmg_set > 5 then
        renderer.text(x, y - 20, 255, 255, 255, dmg_set, "c", 0, "Right click to reset to original position")
    end

    renderer.rec_outline(x, y, 25, 15, 5, 1, {200, 200, 200, dmg_alpha})
    drag_dmg:drag(25, 15)
end

local function mouse_in_bounds(x, y, width, height)
    local mouse_x, mouse_y = ui.mouse_position()
    return mouse_x >= x and mouse_x <= x + width and mouse_y >= y and mouse_y <= y + height
end

local velocity_alpha = 0
local velocity_amount = 0
local vel_drag_alpha = 0
local vel_text_alpha = 0
local drag_velocity = dragging.new("example_dragging", center[1]-50, screen[2] / 3)

function visuals_helpers.velocity_ind()
    local lp = entity.get_local_player()
    if lp == nil then return end
    local r, g, b, a = lua_items.visuals.velocity_warning:get_color()
    local vel_mod = entity.get_prop(lp, 'm_flVelocityModifier')
    local x, y = drag_velocity:get()
    vel_drag_alpha = math.lerp(vel_drag_alpha, ui.is_menu_open() and 155 or 0, 0.05)
    vel_text_alpha = math.lerp(vel_text_alpha, mouse_in_bounds(x, y, 100, 50) and ui.is_menu_open() and 155 or 0, 0.05)
    
    if not ui.is_menu_open() then
        velocity_alpha = math.lerp(velocity_alpha, vel_mod < 1 and 255 or 0, 0.05)
        velocity_amount = math.lerp(velocity_amount, vel_mod, 0.05)
    else
        velocity_alpha = math.lerp(velocity_alpha, 255, 0.05)
        velocity_amount = globals.tickcount() % 50 / 100 * 2
    end

    if velocity_alpha < 5 then return end

    if mouse_in_bounds(x, y, 100, 50) and client.key_state(0x02) == true then
        drag_velocity:set(center[1] - 50, y)
    end

    renderer.text(x + 50, y + 15, 255, 255, 255, velocity_alpha, "c", 0, "- velocity -")
    renderer.glow_module(x + 10 - math.floor(40 * velocity_amount) + 40, y + 30, math.floor(80 * velocity_amount), 3, 6, 3, {r, g, b, velocity_alpha / 2}, {r, g, b, velocity_alpha})

    if vel_text_alpha > 5 then
        renderer.text(x + 50, y - 20, 255, 255, 255, vel_text_alpha, "c", 0, "Click Right Button To Centering")
    end
    renderer.rec_outline(x, y, 100, 50, 5, 1, {200, 200, 200, vel_drag_alpha})

    drag_velocity:drag(100, 50)
end

local defensive_alpha = 0
local defensive_amount = 0
local def_drag_alpha = 0
local def_text_alpha = 0
local drag_defensive = dragging.new("defensive_dragging", center[1]-50, screen[2] / 4)

function visuals_helpers.defensive_ind()
    local lp = entity.get_local_player()
    if lp == nil then return end
    local r, g, b, a = lua_items.visuals.defensive_warning:get_color()

    local vel_mod = is_defensive_active()
    local x, y = drag_defensive:get()

    def_drag_alpha = math.lerp(def_drag_alpha, ui.is_menu_open() and 155 or 0, 0.05)
    def_text_alpha = math.lerp(def_text_alpha, mouse_in_bounds(x, y, 100, 50) and ui.is_menu_open() and 155 or 0, 0.05)
    
    if not ui.is_menu_open() then
        defensive_alpha = math.lerp(defensive_alpha, vel_mod and 255 or 0, 0.05)
        defensive_amount = math.lerp(defensive_amount, vel_mod and 1 or 0, 0.05)
    else
        defensive_alpha = math.lerp(defensive_alpha, 255, 0.05)
        defensive_amount = globals.tickcount() % 50 / 100 * 2
    end

    if defensive_alpha < 5 then return end

    if mouse_in_bounds(x, y, 100, 50) and client.key_state(0x02) == true then
        drag_defensive:set(center[1] - 50, y)
    end

    renderer.text(x + 50, y + 15, 255, 255, 255, defensive_alpha, "c", 0, "- defensive -")
    renderer.glow_module(x + 10 - math.floor(40 * defensive_amount) + 40, y + 30, math.floor(80 * defensive_amount), 3, 6, 3, {r, g, b, defensive_alpha / 2}, {r, g, b, defensive_alpha})

    if def_text_alpha > 5 then
        renderer.text(x + 50, y - 20, 255, 255, 255, def_text_alpha, "c", 0, "Click Right Button To Centering")
    end
    renderer.rec_outline(x, y, 100, 50, 5, 1, {200, 200, 200, def_drag_alpha})

    drag_defensive:drag(100, 50)
end

local debug_alpha = 0
local exp_charged = 0
local debug_state = 'GLOBAL'

function visuals_helpers.debug_panel()
    local lp = entity.get_local_player()
    if lp == nil then return end

    if id == 1 then debug_state = 'GLOBAL'
    elseif id == 2 then debug_state = 'STANDING'
    elseif id == 3 then debug_state = 'MOVING'
    elseif id == 4 then debug_state = 'WALKING'
    elseif id == 5 then debug_state = 'CROUCHING'
    elseif id == 6 then debug_state = 'CROUCHING'
    elseif id == 7 then debug_state = 'AEROBIC'
    elseif id == 8 then debug_state = 'AEROBIC'
    elseif id == 9 then debug_state = 'DISCHARGED'
    elseif id == 10 then debug_state = 'LEGIT AA' end

    exp_charged = math.lerp(exp_charged, check_charge() and 14 or 0, 0.1)

    local r,g,b,a = lua_items.visuals.debug_panel:get_color()

    if lua_items.visuals.debug_panel_style:get() == 'Default' then
        renderer.text(20, screen[2] / 2 - 70, 255, 255, 255, 255, '-', 0, 'ASAKA   ['..string.upper(build)..']   |   '..string.upper(string.sub(username, 1, 11))..'   |   '..math.floor(client.latency()*1000)..' MS')
        local textsize = renderer.measure_text("-", 'ASAKA   ['..string.upper(build)..']   |   '..string.upper(string.sub(username, 1, 11))..'   |   '..math.floor(client.latency()*1000)..' MS')
        renderer.gradient(20, screen[2] / 2 - 55, textsize/2, 1, r,g,b, 50, r,g,b, 255, true)
        renderer.gradient(20 + textsize/2,screen[2] / 2 - 55, textsize/2, 1, r,g,b, 255, r,g,b, 50, true)
        renderer.text(20, screen[2] / 2 - 50, 255, 255, 255, 255, '-', 0, 'CONDITION:  '..debug_state)
        renderer.text(20, screen[2] / 2 - 40, 255, 255, 255, 255, '-', 0, 'TARGET:  ' .. (client.current_threat() and string.upper(string.sub(entity.get_player_name(client.current_threat()), 1, 14)) or 'NIL'))
        renderer.text(20, screen[2] / 2 - 30, 255, 255, 255, 255, '-', 0, 'EXPLOIT  CHARGE:  '..math.floor(exp_charged))
        renderer.text(20, screen[2] / 2 - 20, 255, 255, 255, 255, '-', 0, 'DESYNC:  '..math.abs(math.floor(antiaim_funcs.get_desync())))
    else
        renderer.glow_module(15, screen[2] / 2 - 73, 140, 57, 5, 3, {r,g,b,a}, {20, 20, 20, 75})
        renderer.text(20, screen[2] / 2 - 70, 255, 255, 255, 255, '', 0, '>>  Asaka.lua')
        renderer.text(20, screen[2] / 2 - 60, 255, 255, 255, 255, '', 0, '>>  user: '..string.sub(username, 1, 11))
        renderer.text(20, screen[2] / 2 - 50, 255, 255, 255, 255, '', 0, '>>  build: '..build)
        renderer.text(20, screen[2] / 2 - 40, 255, 255, 255, 255, '', 0, '>>  target: '.. (client.current_threat() and string.sub(entity.get_player_name(client.current_threat()), 1, 10) or 'nil'))
        renderer.text(20, screen[2] / 2 - 30, 255, 255, 255, 255, '', 0, '>>  exploit charge: '..math.floor(exp_charged))
    end
end

local scope_alpha = 0
local s_length = 0
function visuals_helpers.custom_scope()
    local lp = entity.get_local_player()
    if not lp then return end
    if not entity.is_alive(lp) then return end
    local scpd = entity.get_prop(lp, "m_bIsScoped") == 1

    local r,g,b,a = lua_items.visuals.custom_scope:get_color()
    scope_alpha = math.lerp(scope_alpha, scpd and a or 0, 0.10)

    local s_offset = lua_items.visuals.custom_scope_offset:get()
    if scpd then
        s_length = math.lerp(s_length, scpd and lua_items.visuals.custom_scope_length:get() or 0, 0.05)
    else
        s_length = 0
    end

    if scpd then
        renderer.gradient(center[1] + s_offset, center[2], s_length, 1, r,g,b, scope_alpha, r,g,b, 0, true)
        renderer.gradient(center[1] - s_offset+2, center[2], -s_length, 1, r,g,b, scope_alpha, r,g,b, 0, true)

        renderer.gradient(center[1], center[2] + s_offset, 1, s_length, r,g,b, scope_alpha, r,g,b, 0, false)
        renderer.gradient(center[1], center[2] - s_offset+3, 1, -s_length, r,g,b, scope_alpha, r,g,b, 0, false)
    end
end

local queue = {}

function visuals_helpers.kibit_mark()
    local r,g,b,a = lua_items.visuals.kibit_hitmarker:get_color()
    for tick, data in pairs(queue) do
        if globals.curtime() <= data[4] then
            local x1, y1 = renderer.world_to_screen(data[1], data[2], data[3])
            if x1 ~= nil and y1 ~= nil then
                renderer.line(x1 - 6,y1,x1 + 6,y1, r,g,b,a)
                renderer.line(x1,y1 - 6,x1,y1 + 6 , r,g,b,a)
            end
        end
    end
end

local queue2 = {}

client.set_event_callback("bullet_impact", function(e)
    if not lua_items.visuals.bullet_tracers:get() then return end
    if client.userid_to_entindex(e.userid) ~= entity.get_local_player() then return end
    local lx, ly, lz = client.eye_position()
    queue2[globals.tickcount()] = {lx, ly, lz, e.x, e.y, e.z, globals.curtime() + 2}
end)

function visuals_helpers.bullet_tracers()
    local r,g,b,a = lua_items.visuals.bullet_tracers:get_color()
    for tick, data in pairs(queue2) do
        if globals.curtime() <= data[7] then
            local x1, y1 = renderer.world_to_screen(data[1], data[2], data[3])
            local x2, y2 = renderer.world_to_screen(data[4], data[5], data[6])
            if x1 ~= nil and x2 ~= nil and y1 ~= nil and y2 ~= nil then
                renderer.line(x1, y1, x2, y2, r,g,b,a)
            end
        end
    end
end

local arrows_offset = 0

function visuals_helpers.manual_arrows()
    local lp = entity.get_local_player()
    if lp == nil then return end

    local scpd = entity.get_prop(lp, "m_bIsScoped") == 1
    arrows_offset = math.lerp(arrows_offset, scpd and 20 or 0, 0.30)

    if lua_items.visuals.manual_active:get() and yaw_keybinds == 0 then return end


    local r,g,b,a = 40,40,40,150
    local r1,g1,b1,a1 = lua_items.visuals.manual_active:get_color()
    local r2,g2,b2,a2 = lua_items.visuals.manual_ind:get_color()

    local x_offset = 40
    local rect_size = 2
    local width = 10
    local height = 8

    local left_x = center[1] - x_offset - (rect_size + 2)
    local right_x = center[1] + x_offset + (rect_size + 2)

    if lua_items.visuals.manual_ind_type:get() == 'TS' then
        if yaw_keybinds == -90 then
            renderer.triangle(left_x - width, center[2] - arrows_offset, left_x, center[2] - arrows_offset - height, left_x, center[2] + height - arrows_offset, r1,g1,b1,255)
            renderer.triangle(right_x + width, center[2] - arrows_offset, right_x, center[2] - arrows_offset - height, right_x, center[2] + height - arrows_offset, r,g,b,a)
        elseif yaw_keybinds == 90 then
            renderer.triangle(left_x - width, center[2] - arrows_offset, left_x, center[2] - arrows_offset - height, left_x, center[2] + height - arrows_offset, r,g,b,a)
            renderer.triangle(right_x + width, center[2] - arrows_offset, right_x, center[2] - arrows_offset - height, right_x, center[2] + height - arrows_offset, r1,g1,b1,255)
        else
            renderer.triangle(left_x - width, center[2] - arrows_offset, left_x, center[2] - arrows_offset - height, left_x, center[2] + height - arrows_offset, r,g,b,a)
            renderer.triangle(right_x + width, center[2] - arrows_offset, right_x, center[2] - arrows_offset - height, right_x, center[2] + height - arrows_offset, r,g,b,a)
        end

        if entity.get_prop(lp, 'm_flPoseParameter', 11) * 120 - 60 > 0 then
            renderer.rectangle(left_x + rect_size + 2, center[2] - arrows_offset - height, -rect_size, height * 2, r,g,b,a)
            renderer.rectangle(right_x - rect_size - 2, center[2] - arrows_offset - height, rect_size, height * 2, r2,g2,b2,200)
        else
            renderer.rectangle(left_x + rect_size + 2, center[2] - arrows_offset - height, -rect_size, height * 2, r2,g2,b2,200)
            renderer.rectangle(right_x - rect_size - 2, center[2] - arrows_offset - height, rect_size, height * 2, r,g,b,a)
        end
    else
        if yaw_keybinds == -90 then
            renderer.text(left_x, center[2] - arrows_offset, r1,g1,b1,255, 'c+', 0, '<')
            renderer.text(right_x, center[2] - arrows_offset, r2,g2,b2,255, 'c+', 0, '>')
        elseif yaw_keybinds == 90 then
            renderer.text(left_x, center[2] - arrows_offset, r2,g2,b2,255, 'c+', 0, '<')
            renderer.text(right_x, center[2] - arrows_offset, r1,g1,b1,255, 'c+', 0, '>')
        else
            renderer.text(left_x, center[2] - arrows_offset, r2,g2,b2,255, 'c+', 0, '<')
            renderer.text(right_x, center[2] - arrows_offset, r2,g2,b2,255, 'c+', 0, '>')
        end
    end
end

local console_logs = {}

console_logs.add = function(...)
    args = { ... }
    len = #args
    for i = 1, len do
        arg = args[i]
        r, g, b = unpack(arg)

        msg = {}

        if #arg == 3 then
            _G.table.insert(msg, " ")
        else
            for i = 4, #arg do
                _G.table.insert(msg, arg[i])
            end
        end
        msg = _G.table.concat(msg)

        if len > i then
            msg = msg .. "\0"
        end

        client.color_log(r, g, b, msg)
    end
end

console_logs.hitboxes = {"generic", "head", "chest", "stomach", "left arm", "right arm", "left leg", "right leg", "neck", "?", "gear"}

console_logs.on_aim_fire = function(e)
	console_logs[e.id] = {
        original = e,
		history = globals.tickcount() - e.tick
	}
    queue[globals.tickcount()] = {e.x,e.y,e.z, globals.curtime() + 2}
end

console_logs.on_aim_hit = function(e)
    if entity.get_prop(e.target, 'm_iHealth') <= 0 then
        user_info[1] = user_info[1] + 1
        lua_items.main.labelenem:set(' \v\r  Enemies Eliminated: \v'..user_info[1] or '0')
    end

    if not lua_items.misc.ragebot_logs:get() then return end
    if not lua_items.misc.ragebot_logs_select:get('Console') then return end

	if console_logs[e.id] == nil then return end
	local info = 
	{
        type = math.max(0, entity.get_prop(e.target, 'm_iHealth')) > 0,
		prefix = { lua_items.misc.ragebot_hit:get() },
		hit = { lua_items.misc.ragebot_hit:get() },
		name = entity.get_player_name(e.target),
		hitgroup = console_logs.hitboxes[e.hitgroup + 1] or '?',
		aimed_hitgroup = console_logs.hitboxes[console_logs[e.id].original.hitgroup + 1] or '?',
		aimed_hitchance = string.format('%d%%', math.floor(console_logs[e.id].original.hit_chance + 0.5)),
		hp = math.max(0, entity.get_prop(e.target, 'm_iHealth')),

	}

	console_logs.add(
        { info.prefix[1], info.prefix[2], info.prefix[3], '[Asaka] '}, 
		{ 200, 200, 200, 'damaged '}, 
		{ info.hit[1], info.hit[2], info.hit[3],  info.name }, 
		{ 200, 200, 200, ' in the ' }, 
		{ info.hit[1], info.hit[2], info.hit[3], info.hitgroup }, 
		{ 200, 200, 200, info.type and info.hitgroup ~= info.aimed_hitgroup and ' (' or ''},
		{ info.hit[1], info.hit[2], info.hit[3], info.type and (info.hitgroup ~= info.aimed_hitgroup and info.aimed_hitgroup) or '' },
		{ 200, 200, 200, info.type and info.hitgroup ~= info.aimed_hitgroup and ')' or ''},
		{ 200, 200, 200, ' for '},
		{ info.hit[1], info.hit[2], info.hit[3], e.damage},
		{ 200, 200, 200, info.type and e.damage ~= console_logs[e.id].original.damage and ' (' or ''},
		{ info.hit[1], info.hit[2], info.hit[3], info.type and (e.damage ~= console_logs[e.id].original.damage and console_logs[e.id].original.damage) or '' },
		{ 200, 200, 200, info.type and e.damage ~= console_logs[e.id].original.damage and ')' or ''},
		{ 200, 200, 200, info.type and ' damage' or '' },
		{ 200, 200, 200, info.type and ' (' or '' }, { info.hit[1], info.hit[2], info.hit[3], info.type and info.hp or '' }, 
        { 200, 200, 200, info.type and ' hp remaning)' or '' },
		{ 200, 200, 200, ' [hc: ' }, { info.hit[1], info.hit[2], info.hit[3], info.aimed_hitchance },
		{ 200, 200, 200, ' | bt: ' }, { info.hit[1], info.hit[2], info.hit[3], console_logs[e.id].history },
		{ 200, 200, 200, ']' }
    )
end

console_logs.on_aim_miss = function(e)
    if not lua_items.misc.ragebot_logs:get() then return end
    if not lua_items.misc.ragebot_logs_select:get('Console') then return end
    local me = entity.get_local_player()
    local info = {
        prefix = {lua_items.misc.ragebot_miss:get()},
        hit = {lua_items.misc.ragebot_miss:get()},
        name = entity.get_player_name(e.target),
        hitgroup = console_logs.hitboxes[e.hitgroup + 1] or '?',
        aimed_hitgroup = console_logs.hitboxes[console_logs[e.id].original.hitgroup + 1] or '?',
        aimed_hitchance = string.format('%d%%', math.floor(console_logs[e.id].original.hit_chance + 0.5)),
        hp = math.max(0, entity.get_prop(e.target, 'm_iHealth')),
        reason = e.reason,
    }

    console_logs.add(
        {info.prefix[1], info.prefix[2], info.prefix[3], '[Asaka] '}, 
        {200, 200, 200, 'missed shot at '}, {info.hit[1], info.hit[2], info.hit[3], info.name}, 
        {200, 200, 200, ' in the '}, {info.hit[1], info.hit[2], info.hit[3], info.hitgroup}, 
        {200, 200, 200, ' due to '}, {info.hit[1], info.hit[2], info.hit[3], info.reason},
        {200, 200, 200, ' [hc: '}, {info.hit[1], info.hit[2], info.hit[3], info.aimed_hitchance}, 
        {200, 200, 200, ' | bt: '}, {info.hit[1], info.hit[2], info.hit[3], console_logs[e.id].history},
        {200, 200, 200, ']'}
    )
end

local logs = {}
function misc_helpers.ragebot_logs()
    local offset, x, y = 0, screen[1] / 2, screen[2] * 0.9
    for idx, data in ipairs(logs) do
        if (((globals.realtime()/2) * 2.0) - data[3]) < 4.0 and not (#logs > 6 and idx < #logs - 5) then
            data[2] = math.lerp(data[2], 255, 0.15)
        else
            data[2] = math.lerp(data[2], 0, 0.15)
        end
        offset = offset + 45 * (data[2] / 255)

        local r, g, b = unpack(data[4])

        local to_hex = rgba_to_hex(r, g, b, data[2])
        local res_hex = rgba_to_hex(255, 255, 255, data[2])
        text_size_x, text_sise_y = renderer.measure_text("", ' \a'..to_hex..'   \r'..data[1]..' ')
        emo_size_x, emo_size_y = renderer.measure_text("", ' \a'..to_hex..'   \r')


        renderer.glow_module(x - 7 - text_size_x / 2, y - offset-4, text_size_x + 13, 23, 8, 6, {r, g, b, data[2]/3}, {20, 20, 20, data[2]})

        renderer.text(x - 1 - text_size_x / 2, y - offset+1, 255, 255, 255, data[2], "", 0, ' \a'..to_hex..'   ')
        renderer.text(x - 1 - text_size_x / 2 + emo_size_x, y - offset+2, 255, 255, 255, data[2], "", 0, data[1])
        if data[2] < 0.1 or not entity.get_local_player() then table.remove(logs, idx) end
    end
end

renderer.log = function(text, color)
    table.insert(logs, { text, 0, ((globals.realtime() / 2) * 2.0), color})
end

local hitgroup_names = {'generic', 'head', 'chest', 'stomach', 'left arm', 'right arm', 'left leg', 'right leg', 'neck', '?', 'gear'}

local function aim_hit(e)
    if not lua_items.misc.ragebot_logs:get() then return end
    if not lua_items.misc.ragebot_logs_select:get('Screen') then return end
    local group = hitgroup_names[e.hitgroup + 1] or '?'
    renderer.log(string.format('Hit %s in the %s for %d damage', entity.get_player_name(e.target) or "Lopgy", group, e.damage or 0), {lua_items.misc.ragebot_hit:get()})
end


local function aim_miss(e)
    if not lua_items.misc.ragebot_logs:get() then return end
    if not lua_items.misc.ragebot_logs_select:get('Screen') then return end
    local group = hitgroup_names[e.hitgroup + 1] or '?'
    renderer.log(string.format('Missed %s in the %s due to %s', entity.get_player_name(e.target) or "Lopgy", group, e.reason or "?"), {lua_items.misc.ragebot_miss:get()})
end

local fd_reset = false

function misc_helpers.allow_duck(cmd)
    local lp = entity.get_local_player()
    if lp == nil then return end
    local state = cmd.in_duck == 1
    local flags = entity.get_prop(lp, 'm_fFlags')
    local jumpcheck = bit.band(flags, 1) == 0

    if jumpcheck or cmd.in_jump == 1 then
        state = false
    end

    local active, mode = ui.get(ref.fakeduck)

    if active and state then
        local mode_new = 'Off hotkey'

        if mode == 2 or mode == 3 then
            mode_new = 'On hotkey'
        end

        override.set(ref.fakeduck, mode_new)
        fd_reset = false
    elseif not state then
        if not fd_reset then
            override.unset(ref.fakeduck)
            fd_reset = true
        end
    end
end

local tp_state = 'nil'

function misc_helpers.auto_tp(cmd)
    local lp = entity.get_local_player()
    if lp == nil then return end
    local flags = entity.get_prop(lp, 'm_fFlags')
    local jumpcheck = bit.band(flags, 1) == 0
    if is_vulnerable() and jumpcheck then
        tp_state = 'TP'
        cmd.force_defensive = true
        cmd.discharge_pending = true
    elseif not check_charge() then
        tp_state = 'CH'
    else
        tp_state = 'ACT'
    end
end


function misc_helpers.airlag(cmd)
    local lp = entity.get_local_player()
    if lp == nil then return end
    local flags = entity.get_prop(lp, 'm_fFlags')
    local jumpcheck = bit.band(flags, 1) == 0
    if not jumpcheck then return end
    if globals.tickcount() % 10 > 1 then
        cmd.discharge_pending = true
    end
end

local is_hittable = false
local charge_state = false

function misc_helpers.unsafe_charge(cmd)
    local lp = entity.get_local_player()
    if not lp or not entity.is_alive(lp) then return end

    local threat = client.current_threat()
    local in_air = bit.band(entity.get_prop(lp, 'm_fFlags'), 1) == 0
    
    if threat then
        is_hittable = bit.band(entity.get_esp_data(threat).flags, bit.lshift(1, 11)) == 2048
    else
        is_hittable = false
    end

    if is_hittable and not check_charge() and ui.get(ref.doubletap.main[1]) and ui.get(ref.doubletap.main[2]) and in_air then
        ui.set(ref.aimbot, false)
        charge_state = true
    else
        ui.set(ref.aimbot, true)
        charge_state = false
    end
end


function misc_helpers.translate_menu(item)
    if item == "Auto" then
        return "scar20"
    elseif item == "Scout" then
        return "ssg08"
    elseif item == "Awp" then
        return "awp"
    elseif item == "Primary rifle" then
        return "ak47"
    elseif item == "Scoped rifle" then
        return "sg556"
    elseif item == "Machine gun" then
        return "negev"
    elseif item == "Default pistol" then
        return "glock"
    elseif item == "P250" then
        return "p250"
    elseif item == "Dual Berettas" then
        return "elite"
    elseif item == "Light pistol" then
        return "tec9"
    elseif item == "Heavy pistol" then
        return "deagle"
    elseif item == "Grenade" then
        return "hegrenade"
    elseif item == "Smoke" then
        return "smokegrenade"
    elseif item == "Incendiary" then
        return "molotov"
    elseif item == "Flashbang" then
        return "flashbang"
    elseif item == "Kevlar + Helmet" then
        return "vesthelm"
    elseif item == "Defuse kit" then
        return "defuser"
    elseif item == "Zeus" then
        return "Taser"
    else
        client.console_log("Unknown item: ", item)
    end
end

function misc_helpers.buybot()
    if not lua_items.misc.buy_bot:get() then return end
    local primary_weapon = lua_items.misc.buybot_primary:get()
    local secondary_weapon = lua_items.misc.buybot_secondary:get()
    local utility = lua_items.misc.buybot_utility:get()
    local current_buy = nil

    if primary_weapon == "Auto" and entity.get_classname(entity.get_player_weapon(entity.get_local_player())) == "CWeaponSCAR20" then
        current_buy = "buy "..misc_helpers.translate_menu(secondary_weapon)
    else
        current_buy = "buy "..misc_helpers.translate_menu(primary_weapon).."; ".."buy "..misc_helpers.translate_menu(secondary_weapon) 
    end
    if #utility == 0 then
        client.exec(current_buy)
    else
        for i = 1, #utility do
            current_buy = current_buy.."; buy "..misc_helpers.translate_menu(utility[i])
        end
        client.exec(current_buy)
    end
end

local view_cvars = {
    viewmodel_offset_x = cvar.viewmodel_offset_x:get_float(),
    viewmodel_offset_y = cvar.viewmodel_offset_y:get_float(),
    viewmodel_offset_z = cvar.viewmodel_offset_z:get_float(),

}

local scpd_value_x = view_cvars.viewmodel_offset_x
local scpd_value_y = view_cvars.viewmodel_offset_y
local scpd_value_z = view_cvars.viewmodel_offset_z

function misc_helpers.cs2_view()
    local lp = entity.get_local_player()
    if not lp then return end
    if not entity.is_alive(lp) then return end
    local scpd = entity.get_prop(lp, "m_bIsScoped") == 1 
    scpd_value_x = math.lerp(scpd_value_x, scpd and -2 or view_cvars.viewmodel_offset_x, 1)
    scpd_value_y = math.lerp(scpd_value_y, scpd and -8 or view_cvars.viewmodel_offset_y, 1)
    scpd_value_z = math.lerp(scpd_value_z, scpd and -4 or view_cvars.viewmodel_offset_z, 1)
    cvar.sv_competitive_minspec:set_int(0)
    cvar.viewmodel_offset_x:set_float(scpd_value_x)
    cvar.viewmodel_offset_y:set_float(scpd_value_y)
    cvar.viewmodel_offset_z:set_float(scpd_value_z)
    cvar.fov_cs_debug:set_int(scpd and 90 or 0)
end

lua_items.misc.cs2_view:set_callback(function(self)
    cvar.viewmodel_offset_x:set_float(view_cvars.viewmodel_offset_x)
    cvar.viewmodel_offset_y:set_float(view_cvars.viewmodel_offset_y)
    cvar.viewmodel_offset_z:set_float(view_cvars.viewmodel_offset_z)
    cvar.fov_cs_debug:set_int(0)
end)

local phrases = {
    "✧ ｉｎｆｉｎｉｘ ｏｖｅｒｃｌｏｃｋ ｔｈｅ ｃｏｄｅ ✧",
    "⚡ ｉｎｆｉｎｉｘ.ｌｕａ ｍｅｌｔｓ ｔｈｅ ｍａｔｒｉｘ ⚡",
    "♛ ｉｎｆｉｎｉｘ ｉｓ ｔｈｅ ｖｏｉｄ ｏｆ ｐｏｗｅｒ ♛",
    "⌘ ｉｎｆｉｎｉｘ ｃｏｄｅ ｉｓ ｔｈｅ ｆｕｔｕｒｅ ⌘",
    "∞ ｉｎｆｉｎｉｘ ｖｏｒｔｅｘ ｉｓ ｕｎｓｔｏｐｐａｂｌｅ ∞",
    "𖡆 ｉｎｆｉｎｉｘ ｈａｃｋｅｄ ｔｈｅ ｔｉｍｅｌｉｎｅ 𖡆",
    "꧁ ｉｎｆｉｎｉｘ ｉｓ ｔｈｅ ｄｒａｋｅ ｏｆ ｃｏｄｅ ꧂",
    "⚔️ ｉｎｆｉｎｉｘ ｂｒｅａｋｓ ｔｈｅ ｓｙｓｔｅｍ ⚔️",
    "♦ ｉｎｆｉｎｉｘ ｉｓ ｔｈｅ ｄｅｖｉｌ'ｓ ｃｏｍｐｉｌｅｒ ♦",
    "♡ ｉｎｆｉｎｉｘ ｍａｋｅｓ ｔｈｅ ｗｏｒｌｄ ｂｅｔｔｅｒ ♡",
    "✨ ｉｎｆｉｎｉｘ ｉｓ ｔｈｅ ｇｌｉｔｃｈ ｉｎ ｔｈｅ ｍａｔｒｉｘ ✨",
    "⇌ ｉｎｆｉｎｉｘ ｉｓ ｔｈｅ ｂｒｉｄｇｅ ｂｅｔｗｅｅｎ ｒｅａｌｉｔｉｅｓ ⇌",
    "⟁ ｉｎｆｉｎｉｘ ｉｓ ｔｈｅ ｃｏｄｅ ｏｆ ｔｈｅ ｇｏｄｓ ⟁",
    "⌘ ｉｎｆｉｎｉｘ ｉｓ ｔｈｅ ｋｅｙ ｔｏ ｔｈｅ ｕｎｉｖｅｒｓｅ ⌘",
    "𖹭 ｉｎｆｉｎｉｘ ｉｓ ｔｈｅ ｄｒｅａｍ ｏｆ ｔｈｅ ｆｕｔｕｒｅ 𖹭",
    "✦ ｉｎｆｉｎｉｘ ｉｓ ｔｈｅ ｓｏｕｌ ｏｆ ｔｈｅ ｍａｃｈｉｎｅ ✦",
    "♛ ｉｎｆｉｎｉｘ ｉｓ ｔｈｅ ｋｉｎｇ ｏｆ ｔｈｅ ｃｏｄｅ ♛",
    "⚡ ｉｎｆｉｎｉｘ ｉｓ ｔｈｅ ｓｐａｒｋ ｏｆ ｔｈｅ ｆｕｔｕｒｅ ⚡",
    "∞ ｉｎｆｉｎｉｘ ｉｓ ｔｈｅ ｅｎｄｌｅｓｓ ｌｏｏｐ ∞",
    "𖡆 ｉｎｆｉｎｉｘ ｉｓ ｔｈｅ ｈａｃｋ ｔｈａｔ ｂｒｅａｋｓ ｔｈｅ ｒｕｌｅｓ 𖡆",
}

function misc_helpers.fastladder(e)
    local local_player = entity.get_local_player()
    local pitch, yaw = client.camera_angles()
    if entity.get_prop(local_player, "m_MoveType") == 9 then
        e.yaw = math.floor(e.yaw+0.5)
        e.roll = 0
        if e.forwardmove == 0 then
            if e.sidemove ~= 0 then
                e.pitch = 89
                e.yaw = e.yaw + 180
                if e.sidemove < 0 then
                    e.in_moveleft = 0
                    e.in_moveright = 1
                end
                if e.sidemove > 0 then
                    e.in_moveleft = 1
                    e.in_moveright = 0
                end
            end
        end
        if e.forwardmove > 0 then
            if pitch < 45 then
                e.pitch = 89
                e.in_moveright = 1
                e.in_moveleft = 0
                e.in_forward = 0
                e.in_back = 1
                if e.sidemove == 0 then
                    e.yaw = e.yaw + 90
                end
                if e.sidemove < 0 then
                    e.yaw = e.yaw + 150
                end
                if e.sidemove > 0 then
                    e.yaw = e.yaw + 30
                end
            end 
        end
        if e.forwardmove < 0 then
            e.pitch = 89
            e.in_moveleft = 1
            e.in_moveright = 0
            e.in_forward = 1
            e.in_back = 0
            if e.sidemove == 0 then
                e.yaw = e.yaw + 90
            end
            if e.sidemove > 0 then
                e.yaw = e.yaw + 150
            end
            if e.sidemove < 0 then
                e.yaw = e.yaw + 30
            end
        end
    end
end

local fps_cvars = {
    r_3dsky = cvar.r_3dsky:get_int(),
    fog_enable = cvar.fog_enable:get_int(),
    fog_enable_water_fog = cvar.fog_enable_water_fog:get_int(),
    fog_enableskybox = cvar.fog_enableskybox:get_int(),
    r_shadows = cvar.r_shadows:get_int(),
    violence_hblood = cvar.violence_hblood:get_int(),
    violence_ablood = cvar.violence_ablood:get_int(),
    r_decals = cvar.r_decals:get_int(),
    mat_postprocess_enable = cvar.mat_postprocess_enable:get_int(),
    cl_disable_ragdolls = cvar.cl_disable_ragdolls:get_int(),
    r_eyegloss = cvar.r_eyegloss:get_int(),
    r_eyemove = cvar.r_eyemove:get_int(),
    r_eyeshift_x = cvar.r_eyeshift_x:get_int(),
    r_eyeshift_y = cvar.r_eyeshift_y:get_int(),
    r_eyeshift_z = cvar.r_eyeshift_z:get_int(),
    r_eyesize = cvar.r_eyesize:get_int(),
    cl_detail_avoid_radius = cvar.cl_detail_avoid_radius:get_int(),
    cl_detail_max_sway = cvar.cl_detail_max_sway:get_int(),
    dsp_slow_cpu = cvar.dsp_slow_cpu:get_int(),
    func_break_max_pieces = cvar.func_break_max_pieces:get_int(),
    r_drawtracers = cvar.r_drawtracers:get_int(),
    r_dynamic = cvar.r_dynamic:get_int(),
    r_drawparticles = cvar.r_drawparticles:get_int(),
    muzzleflash_light = cvar.muzzleflash_light:get_int(),
    mat_hdr_enabled = cvar.mat_hdr_enabled:get_int(),
}

function misc_helpers.fps_boost(value)
    cvar.r_3dsky:set_int((value and lua_items.misc.fps_opt:get('3D Sky')) and 0 or fps_cvars.r_3dsky)

    cvar.fog_enable:set_int((value and lua_items.misc.fps_opt:get('Fog')) and 0 or fps_cvars.fog_enable)
    cvar.fog_enable_water_fog:set_int((value and lua_items.misc.fps_opt:get('Fog')) and 0 or fps_cvars.fog_enable_water_fog)
    cvar.fog_enableskybox:set_int((value and lua_items.misc.fps_opt:get('Fog')) and 0 or fps_cvars.fog_enableskybox)

    cvar.r_shadows:set_int((value and lua_items.misc.fps_opt:get('Shadows')) and 0 or fps_cvars.r_shadows)

    cvar.violence_hblood:set_int((value and lua_items.misc.fps_opt:get('Blood')) and 0 or fps_cvars.violence_hblood)
    cvar.violence_ablood:set_int((value and lua_items.misc.fps_opt:get('Blood')) and 0 or fps_cvars.violence_ablood)

    cvar.r_decals:set_int((value and lua_items.misc.fps_opt:get('Decals')) and 0 or fps_cvars.r_decals)

    cvar.mat_postprocess_enable:set_int((value and lua_items.misc.fps_opt:get('Bloom')) and 0 or fps_cvars.mat_postprocess_enable)

    cvar.cl_disable_ragdolls:set_int((value and lua_items.misc.fps_opt:get('Ragdols')) and 0 or fps_cvars.cl_disable_ragdolls)

    cvar.r_eyegloss:set_int((value and lua_items.misc.fps_opt:get('Eye candy')) and 0 or fps_cvars.r_eyegloss)
    cvar.r_eyemove:set_int((value and lua_items.misc.fps_opt:get('Eye candy')) and 0 or fps_cvars.r_eyemove)
    cvar.r_eyeshift_x:set_int((value and lua_items.misc.fps_opt:get('Eye candy')) and 0 or fps_cvars.r_eyeshift_x)
    cvar.r_eyeshift_y:set_int((value and lua_items.misc.fps_opt:get('Eye candy')) and 0 or fps_cvars.r_eyeshift_y)
    cvar.r_eyeshift_z:set_int((value and lua_items.misc.fps_opt:get('Eye candy')) and 0 or fps_cvars.r_eyeshift_z)
    cvar.r_eyesize:set_int((value and lua_items.misc.fps_opt:get('Eye candy')) and 0 or fps_cvars.r_eyesize)

    cvar.r_drawparticles:set_int((value and lua_items.misc.fps_opt:get('Molotov')) and 0 or fps_cvars.r_drawparticles)

    cvar.cl_detail_avoid_radius:set_int((value and lua_items.misc.fps_opt:get('Other')) and 0 or fps_cvars.cl_detail_avoid_radius)
    cvar.cl_detail_max_sway:set_int((value and lua_items.misc.fps_opt:get('Other')) and 0 or fps_cvars.cl_detail_max_sway)
    cvar.dsp_slow_cpu:set_int((value and lua_items.misc.fps_opt:get('Other')) and 0 or fps_cvars.dsp_slow_cpu)
    cvar.func_break_max_pieces:set_int((value and lua_items.misc.fps_opt:get('Other')) and 0 or fps_cvars.func_break_max_pieces)
    cvar.r_drawtracers:set_int((value and lua_items.misc.fps_opt:get('Other')) and 0 or fps_cvars.r_drawtracers)
    cvar.r_dynamic:set_int((value and lua_items.misc.fps_opt:get('Other')) and 0 or fps_cvars.r_dynamic)
    cvar.muzzleflash_light:set_int((value and lua_items.misc.fps_opt:get('Other')) and 0 or fps_cvars.muzzleflash_light)
    cvar.mat_hdr_enabled:set_int((value and lua_items.misc.fps_opt:get('Other')) and 0 or fps_cvars.mat_hdr_enabled)
end

lua_items.misc.fps_boost:set_callback(function(self)
    if self:get() and lua_items.misc.fps_always:get() then
        misc_helpers.fps_boost(true)
    else
        misc_helpers.fps_boost(false)
    end
end)

lua_items.misc.fps_always:set_callback(function(self)
    if lua_items.misc.fps_boost:get() and self:get() then
        misc_helpers.fps_boost(true)
    else
        misc_helpers.fps_boost(false)
    end
end)

lua_items.misc.fps_opt:set_callback(function(self)
    if lua_items.misc.fps_boost:get() and lua_items.misc.fps_always:get() then
        misc_helpers.fps_boost(true)
    else
        misc_helpers.fps_boost(false)
    end
end)

function misc_helpers.filter(bool)
    cvar.developer:set_int(0)
    cvar.con_filter_enable:set_int(bool and 1 or 0)
    cvar.con_filter_text:set_string(bool and "IrWL5106TZZKNFPz4P4Gl3pSN?J370f5hi373ZjPg%VOVh6lN" or "")
    client.exec(bool and "con_filter_enable 1" or "con_filter_enable 0")
end

lua_items.misc.filter_console:set_callback(function(self)
    client.exec(self:get() and 'clear')
    misc_helpers.filter(self:get())
end)

function misc_helpers.animated_clantag(text, indices) 
    if not globals.mapname() then return end
    local text_anim = '               ' .. text .. '                      '
    local tickcount = globals.tickcount() + math.floor(client.real_latency()+0.22 / globals.tickinterval() + 0.5)
    local i = tickcount / math.floor(0.3 / globals.tickinterval() + 0.5)
    i = math.floor(i % #indices) i = indices[i+1]+1 
    return string.sub(text_anim, i, i+15)
end

function misc_helpers.set_clan()
    if not globals.mapname() then return end
    if not globals.chokedcommands() then return end
    ui.set(ref.clantag, false)
    local tag_text = misc_helpers.animated_clantag('Asaka.lua', {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 0})
    local game_rules = entity.get_game_rules()
    if not game_rules then return end
	if entity.get_prop(game_rules,'m_gamePhase') == 5 or entity.get_prop(game_rules,'m_gamePhase') == 4 then
		tag_text = misc_helpers.animated_clantag('Asaka.lua', {12})
		client.set_clan_tag(tag_text)
	elseif tag_text ~= tag_text_prev then
		client.set_clan_tag(tag_text)
	end
	tag_text_prev = tag_text
end

lua_items.misc.clantag:set_callback(function(self)
    if not self:get() then
        client.delay_call(0.2, function()
            client.set_clan_tag('')
        end)
    end
end)

local GameStateAPI = panorama.open().GameStateAPI

local lastChatMessage = {}

function misc_helpers.onPlaySay(e)
	local sender = client.userid_to_entindex(e.userid)
	if not entity.is_enemy(sender) then return end

	if GameStateAPI.IsSelectedPlayerMuted(GameStateAPI.GetPlayerXuidStringFromEntIndex(sender)) then return end

	client.delay_call(0.2, function()
		if lastChatMessage[sender] ~= nil and math.abs(globals.realtime() - lastChatMessage[sender]) < 0.4 then
			return
		end


		local enemyTeamName = entity.get_prop(entity.get_player_resource(), "m_iTeam", sender) == 2 and "T" or "CT"

		local placeName = entity.get_prop(sender, "m_szLastPlaceName")
		local enemyName = entity.get_player_name(sender)
		
		local localizeStr = ("Cstrike_Chat_%s_%s"):format(enemyTeamName, entity.is_alive(sender) and "Loc" or "Dead")
		local msg = localize(localizeStr, {
			s1 = enemyName,
			s2 = e.text,
			s3 = localize(placeName ~= "" and placeName or "UI_Unknown")
		})

		chat.print_player(sender, msg)
	end)
end

function misc_helpers.onPlayChat(e)
	if not entity.is_enemy(e.entity) then return end
	lastChatMessage[e.entity] = globals.realtime()
end

local player_data = { }

function rage_helpers.erase_player_data()
    for k in pairs(player_data) do
        player_data[k] = nil
    end
end

function rage_helpers.unset_player_body_yaw(entindex)
    plist.set(entindex, 'Force body yaw', false)
    plist.set(entindex, 'Force body yaw value', 0)
end

function rage_helpers.set_player_body_yaw(entindex, value)
    plist.set(entindex, 'Force body yaw', true)
    plist.set(entindex, 'Force body yaw value', value)
end

function rage_helpers.reset_plist()
    for i = 1, 64 do
        rage_helpers.unset_player_body_yaw(i)
    end
end

function rage_helpers.get_max_desync_delta(animstate)
    local duck_amount = animstate.duck_amount

    local stop_to_full_running_fraction = animstate.stop_to_full_running_fraction

    local speed_fraction = math.max(0, math.min(animstate.feet_speed_forwards_or_sideways, 1))
    local speed_factor = math.max(0, math.min(animstate.feet_speed_unknown_forwards_or_sideways, 1))

    local value = ((stop_to_full_running_fraction * -0.30000001) - 0.19999999) * speed_fraction + 1

    if duck_amount > 0 then
        value = value + ((duck_amount * speed_factor) * (0.5 - value))
    end

    return animstate.max_yaw * value
end
    
function rage_helpers.reset_reso_date(e)
    local me = entity.get_local_player()
    local userid = client.userid_to_entindex(e.userid)

    if me ~= userid then return end

    rage_helpers.erase_player_data()
end


function rage_helpers.jitter_resolver()
    local me = entity.get_local_player()

    if me == nil then return end

    local enemies = entity.get_players(true)

    if enemies == nil then return end

    local my_origin = vector(
        entity.get_origin(me)
    )

    for i = 1, #enemies do
        local enemy = enemies[i]

        local player_info = c_entity.new(enemy)

        if player_info == nil then goto continue end

        if not player_data[enemy] then
            player_data[enemy] = {
            misses = 0,
            last_yaw = 0,
            last_yaw_update_time = 0
            }
        end

        local data = player_data[enemy]

        if data == nil then goto continue end

        local is_correction_active = plist.get(enemy, 'Correction active')

        if not is_correction_active or data.misses > 2 then
            rage_helpers.unset_player_body_yaw(enemy)
            goto continue
        end

        local animstate = player_info:get_anim_state()

        if animstate == nil then goto continue end

        local head_center_position = vector(entity.hitbox_position(enemy, 0))

        local targets = vector((head_center_position - my_origin):angles())
        local yaw = math.normalize(targets.y - animstate.eye_angles_y + 180, -180, 180)

        if data.last_yaw ~= yaw then
            if math.abs(data.last_yaw - yaw) >= 20 and math.abs(data.last_yaw - yaw) <= 340 then
                data.last_yaw_update_time = globals.tickcount() + 15
            end
            data.last_yaw = yaw
        end

        local is_jitter = data.last_yaw_update_time > globals.tickcount()

        if not is_jitter then
            rage_helpers.unset_player_body_yaw(enemy)

            goto continue
        end

        local mod = data.misses == 0 and 1 or -1
        local side = math.clamp(yaw, -1, 1) * mod

        local max_desync = rage_helpers.get_max_desync_delta(animstate)
        rage_helpers.set_player_body_yaw(enemy, max_desync * side)

        ::continue::
    end
end

lua_items.ragebot.resolver:set_callback(function(self)
    if not self:get() then
        rage_helpers.erase_player_data()
        rage_helpers.reset_plist()
    end
end)


function rage_helpers.predict()
    local lp = entity.get_local_player()
    if not lp then return end

    if not entity.is_alive(lp) then return end

    cvar.cl_interp:set_int(client.real_latency()*1000 < 45 and 0)
    cvar.cl_interpolate:set_int(client.real_latency()*1000 < 45 and 1 or 0)
    cvar.cl_interp_ratio:set_int(client.real_latency()*1000 < 45 and 2 or 0)
    cvar.cl_clock_correction:set_int(lua_items.ragebot.pr_over_clor:get() and 0 or 1)
end

function rage_helpers.defensive_fix(cmd)
    local lp = entity.get_local_player()
    if not lp or not entity.is_alive(lp) then return end
    local on_ground = bit.band(entity.get_prop(lp, 'm_fFlags'), 1) == 1
    if on_ground and ui.get(ref.peek[1]) and ui.get(ref.peek[2]) and check_charge() and cmd.quick_stop then
        cmd.in_speed = 1
    end
end

function rage_helpers.air_qs(cmd)
    local lp = entity.get_local_player()
    if not lp then return end
    if not entity.is_alive(lp) then return end

    local ticks = 0
    local players = entity.get_players(true)
    local lpvec = vector(entity.get_prop(lp, "m_vecOrigin"))
    local weapon = entity.get_player_weapon(lp)
    local class = entity.get_classname(weapon)

    if class ~= "CWeaponSSG08" then return end
    local vecvelocity = { entity.get_prop(lp, 'm_vecVelocity') }

    local check_vel = vecvelocity[3] > -20 and vecvelocity[3] < 80
    local flags = entity.get_prop(lp, 'm_fFlags')
    local jumpcheck = bit.band(flags, 1) == 0

    local enemy = client.current_threat()
    if not enemy then return end
    if not jumpcheck then return end
    local enemyvec = vector(entity.get_origin(enemy))
    local trace_l = vector(entity.get_origin(lp))
    if not check_vel then return end
    for i=1, #players do
        if players == nil then return end
        local x1, y1, z1 = entity.get_prop(players[i], "m_vecOrigin")

        local dist = anti_knife_dist(lpvec.x, lpvec.y, lpvec.z, x1, y1, z1)
        if dist <= 1500 then
            if cmd.quick_stop then
                if (globals.tickcount() - ticks) > 3 then
                    cmd.in_speed = 1
                end
            else
                ticks = globals.tickcount()
            end
        end
    end
end

local autopeek = { ui.reference( 'RAGE', 'Other', 'Quick peek assist' ) }
local mindmg = { ui.reference('RAGE', 'Aimbot', 'Minimum damage') }
local mindamage_override = { ui.reference('RAGE', 'Aimbot', 'Minimum damage override') }
local quick_peek_assist = { ui.reference( 'RAGE', 'Other', 'Quick peek assist' ) }

local enemy_lc_data = { }
local function create_new_record( player )
    local data = { }

    data.player = player

    data.last_simtime = 0
    data.origin = vector( entity.get_origin( player ) )

    data.breaking_lc = false
    data.defensive_active_until = 0
    data.defensive = false

    function data.update( )
        data.defensive = bit.band( entity.get_esp_data( data.player ).flags, 131072 ) == 131072
        local simtime = toticks( entity.get_prop( player, 'm_flSimulationTime' ) )
        local origin = vector( entity.get_origin( player ) )

        local delta = simtime - data.last_simtime

        if delta < 0 then
            data.defensive_active_until = globals.tickcount( ) + math.abs( delta )
        else
            data.breaking_lc = ( data.origin - origin ):length2dsqr( ) > 4096
            data.origin = origin
        end

        data.last_simtime = simtime
    end

    enemy_lc_data[ player ] = data

    return data
end

-- god forbid this hardcode shit please dont send me into hell ????
local tick_to_distance = {
    0,
    0.33566926072059,
    0.90550823109139,
    1.7094571925458,
    2.7475758645732,
    4.0198045277169,
    5.5243356897069,
    7.2423273783409,
    9.1564213090631,
    11.250673856852,
    13.510480438002,
    15.922361837797,
    18.473989413581,
    21.153990043142,
    23.951936812474,
    26.858254779359,
    29.864120158319,
    32.961441695549,
    36.142785057665,
    39.401338315411,
    42.730817707458,
    46.125502156263,
    49.580063421207,
    53.08964170921,
    56.649735547569,
    60.256252190999,
    63.905432011078,
    67.59383918326,
    71.318242246617,
    75.075708340563,
    78.863628408227,
    82.67942790961,
    86.520915828495,
    90.385926351936,
    94.272651987509,
    98.17890171902,
    102.08515145053,
    105.99140118205,
    109.89765091356,
    113.80390064508,
    117.7101503766,
    121.61640010812,
    125.52264983965,
    129.42889957117,
    133.3351493027,
    137.24139903422,
    141.14764876575,
    145.05389849727,
    148.9601482288,
    152.86639796033,
    156.77264769186,
    160.67889742339,
    164.58514715492,
    168.49139688645,
    172.39764661798,
    176.30389634951,
    180.21014608104,
    184.11639581258,
    188.02264554411,
    191.92889527564,
    195.83514500718,
    199.74139473871,
    203.64764447024,
    207.55389420178,
}

local function can_hit_in_x_ticks( wanted_pos_distance, max_speed, ticks )
    local distance_mult = max_speed / 250
    local wanted_distance = wanted_pos_distance * distance_mult

    local max_distance = tick_to_distance[ ticks ] * distance_mult

    return wanted_distance <= max_distance
end


local debug_visuals = { }
local visuals = {
    peeking_points = { },
    found_point = nil,
}
local function debug_visualize( positions, name )
    if type( positions ) ~= 'table' then
        positions = { positions }
    end

    debug_visuals[ name ] = positions
end

local function set_visual_peeking_points( points )
    visuals.peeking_points = points
end

local e_hitboxes = {
    [ 'head' ] = 1,
    [ 'stomach' ] = 2,
    [ 'chest' ] = 3,
    [ 'limbs' ] = 4
}

local hitboxes = {
    { 0 },
    { 2, 3, 4 },
    { 5, 6 },
    {
        13, 14, 15, 16, 17, 18, -- arms
        7, 8, 9, 10, -- legs
        11, 1, -- feet
    },
}

local cache = {
    autopeek_position = vector( 0, 0, 0 ),
    last_seen = 0,
    found_position = vector( 0, 0, 0 ),
    found_position_dist = 1,
}

local closest_enemy = nil

local function get_min_dmg( )
    if ui.get( mindamage_override[ 1 ] ) and ui.get( mindamage_override[ 2 ] ) then
        return ui.get( mindamage_override[ 3 ] )
    end

    return ui.get( mindmg[ 1 ] )
end

local function reset_cache( )
    cache.last_seen = 0
    cache.found_position = vector( 0, 0, 0 )
end 

local function get_closest_enemy( )
    local screen_center = vector( client.screen_size( ) ) / 2
   
    local smallest_distance = math.huge
    local closest_enemy_found = nil

    local enemies = entity.get_players( true )
    for i = 1, #enemies do
        local enemy = enemies[ i ]

        if enemy == nil then
            goto continue
        end

        --* only check for enemies that are alive and not dormant
        if not entity.is_alive( enemy ) or entity.is_dormant( enemy ) then
            goto continue
        end
        
        local enemy_position = vector( entity.get_prop( enemy, 'm_vecOrigin' ) )
        
        local x, y = renderer.world_to_screen( enemy_position.x, enemy_position.y, enemy_position.z )

        local enemy_screen = vector( x, y, 0 )
        
        if enemy_screen.x == nil or enemy_screen.y == nil then
            goto continue
        end

        local distance = ( enemy_screen - screen_center ):length( )

        if distance < smallest_distance then
            smallest_distance = distance
            closest_enemy_found = enemy
        end

        ::continue::
    end

    closest_enemy = closest_enemy_found
end

local function get_multipoint( ent, hitbox_center, scale )
    local target_pos = vector( entity.get_prop( ent, 'm_vecOrigin' ) )

    local lp = entity.get_local_player( )
    local lp_pos = vector( entity.get_prop( lp, 'm_vecOrigin' ) )

    local lp_to_tgt = target_pos - lp_pos
    local _, new_yaw = lp_to_tgt:angles( )

    local max_check_dist = 5

    local mp_poses = { }

    for side = -1, 1, 2 do
        local yaw = new_yaw + ( 90 * side )
        local yaw_rad = math.rad( yaw )

        local x = math.cos( yaw_rad )
        local y = math.sin( yaw_rad )

        local mp_start = hitbox_center + vector( x * max_check_dist, y * max_check_dist, 0 )
        local diff = hitbox_center - mp_start

        local frac = client.trace_line( lp, mp_start.x, mp_start.y, mp_start.z, hitbox_center.x, hitbox_center.y, hitbox_center.z )

        local mp_pos = hitbox_center + ( diff * ( 1 - frac ) ) * scale

        table.insert( mp_poses, mp_pos )
    end

    return mp_poses
end

local function get_head_multipoint( ent, hitbox_center, scale )
    local lp = entity.get_local_player( )
    local side_mp = get_multipoint( ent, hitbox_center, scale )

    local max_check_dist = -5
    local mp_start = hitbox_center + vector( 0, 0, max_check_dist )
    local diff = hitbox_center - mp_start

    local frac = client.trace_line( lp, mp_start.x, mp_start.y, mp_start.z, hitbox_center.x, hitbox_center.y, hitbox_center.z )

    local mp_pos = hitbox_center + ( diff * ( 1 - frac ) ) * scale

    table.insert( side_mp, mp_pos )

    return side_mp
end

local function get_player_points( player )
    local points = { }

    local find = {
        head = true,
        chest = true,
        stomach = true,
        limbs = false
    }

    local mp = {
        head = false,
        chest = false,
        stomach = false,
        limbs = false
    }

    local mp_scale = {
        head = 1,
        chest = 1,
        stomach = 1,
        limbs = 0
    }

    if find.head then
        local head_hitboxes = hitboxes[ e_hitboxes[ 'head' ] ]

        for i = 1, #head_hitboxes do
            local head = vector( entity.hitbox_position( player, head_hitboxes[ i ] ) )

            if not mp.head then
                table.insert( points, head )
            else
                local multipoints = get_head_multipoint( player, head, mp_scale.head )

                for mp_idx = 1, #multipoints do
                    table.insert( points, multipoints[ mp_idx ] )
                end
            end
        end
    end

    if find.chest then
        local chest_hitboxes = hitboxes[ e_hitboxes[ 'chest' ] ]
        for i = 1, #chest_hitboxes do
            local chest = vector( entity.hitbox_position( player, chest_hitboxes[ i ] ) )

            if not mp.chest then
                table.insert( points, chest )
            else
                local multipoints = get_multipoint( player, chest, mp_scale.chest )
                
                for mp_idx = 1, #multipoints do
                    table.insert( points, multipoints[ mp_idx ] )
                end
            end
        end
    end

    if find.stomach then
        local stomach_hitboxes = hitboxes[ e_hitboxes[ 'stomach' ] ]
        for i = 1, #stomach_hitboxes do
            local stomach = vector( entity.hitbox_position( player, stomach_hitboxes[ i ] ) )
            
            if not mp.stomach then
                table.insert( points, stomach )
            else
                local multipoints = get_multipoint( player, stomach, mp_scale.stomach )

                for mp_idx = 1, #multipoints do
                    table.insert( points, multipoints[ mp_idx ] )
                end
            end
        end
    end

    if find.limbs then
        local limbs_hitboxes = hitboxes[ e_hitboxes[ 'limbs' ] ]
        for i = 1, #limbs_hitboxes do
            local limb = vector( entity.hitbox_position( player, limbs_hitboxes[ i ] ) )
            table.insert( points, limb )
        end
    end

    return points
end

local function get_peeking_points( lp )
    local lp_origin = vector( entity.get_prop( lp, 'm_vecOrigin' ) )
    local lp_eye = vector( client.eye_position( ) )

    local _pitch, yaw = client.camera_angles( )

    local head_height = lp_eye.z - lp_origin.z

    local start_offset = lua_items.ragebot.dot_offset:get()
    local dots = lua_items.ragebot.dot_amount:get()
    local total_distance = lua_items.ragebot.dot_span:get()
    local gap = total_distance / dots

    local dot_positions = { }

    for i = -1, 1, 2 do
        local dot_yaw = yaw + ( 90 * i )

        local forwardvector = vector( math.cos( math.rad( dot_yaw ) ), math.sin( math.rad( dot_yaw ) ), 0 )

        for dot_iter = 1, dots do
            local dot_position = lp_eye + ( forwardvector * ( gap * dot_iter ) ) + ( forwardvector * start_offset )

            local trace_res = trace.line( dot_position, dot_position + vector( 0, 0, -200 ), { mask = 'MASK_SOLID_BRUSHONLY' } )
	
            local trace_fraction = trace_res.fraction
            if trace_fraction < 1 then
                local end_pos = trace_res.end_pos + vector( 0, 0, head_height )

                if ( end_pos.z - lp_origin.z ) > 40 then
                    dot_iter = dots
                end

                dot_position = end_pos
            end

            local trace_res = trace.line( lp_eye, dot_position, { skip = entity.get_players( ), mask = 'MASK_SOLID' } )
            if trace_res.fraction == 1 then
                table.insert( dot_positions, dot_position )
            else
                local last_dot_pos = lp_eye + ( ( forwardvector * ( gap * dot_iter ) ) + ( forwardvector * start_offset ) ) * trace_res.fraction - forwardvector * 19
                last_dot_pos.z = dot_position.z
                table.insert( dot_positions, last_dot_pos )
                break
            end
        end
    end

    return dot_positions
end

local function can_hit_from_positions( lp, positions, target, target_hitpoints )
    local minimum_damage = get_min_dmg( )

    for i = 1, #positions do
        local position = positions[ i ]
        
        visuals.found_point = i

        for j = 1, #target_hitpoints do
            local hitpoint = target_hitpoints[ j ]
            
            -- client.trace_bullet(from_player, from_x, from_y, from_z, to_x, to_y, to_z, skip_players)
            local hit_entity, simulated_dmg = client.trace_bullet( lp, position.x, position.y, position.z, hitpoint.x, hitpoint.y, hitpoint.z, false )

            local hit_player_name = entity.get_player_name( hit_entity )
            local target_health = entity.get_prop( hit_entity, 'm_iHealth' )

            if hit_entity == target then
                local wanted_dmg = minimum_damage

                if minimum_damage > 100 then
                    wanted_dmg = target_health + ( minimum_damage - 100 )
                end

                -- if mindmg is low enough or we can kill the target with one shot, return true or we can overall shoot the target
                if simulated_dmg >= target_health or simulated_dmg > wanted_dmg then
                    cache.found_position = position
                    cache.found_position_dist = ( cache.autopeek_position - cache.found_position ):length2d( )
                    return true
                end
            elseif hit_entity ~= nil and entity.is_alive( hit_entity ) then
                -- if we hit a different target,check if we can 1 shot it
                local wanted_dmg = minimum_damage

                if minimum_damage > 100 then
                    wanted_dmg = target_health + ( minimum_damage - 100 )
                end


                if simulated_dmg >= target_health or simulated_dmg > wanted_dmg then
                    cache.found_position = position
                    return true
                end
            end
        end
    end

    visuals.found_point = nil
    return false
end

local function ready_to_shoot( lp, cmd )
    local slowdown = entity.get_prop( lp, 'm_flVelocityModifier' ) < 0.9
    local has_user_input = cmd.in_moveleft == 1 or cmd.in_moveright == 1 or cmd.in_back == 1 or cmd.in_forward == 1 or cmd.in_jump == 1
    local wep = entity.get_player_weapon( entity.get_local_player( ) )

    local next_shot_ready = false
    if wep then
        local reloading = entity.get_prop( wep, 'm_bInReload' ) == 1
        local next_attack_ready = entity.get_prop( wep, 'm_flNextPrimaryAttack' ) < globals.curtime( )

        if not reloading and next_attack_ready then
            next_shot_ready = true
        end
    end
    
    local can_normally_shoot = next_shot_ready

    return not ( ( slowdown or has_user_input or not next_shot_ready ) and not can_normally_shoot )
end

local function move_to_pos( cmd, lp, lp_pos, new_pos )
    local distance = lp_pos:dist( new_pos ) + 5
    local unit_vec = ( new_pos - lp_pos ):normalized( )
    
    new_pos = lp_pos + unit_vec * ( distance + 5 )

    if cmd.forwardmove == 0 and cmd.sidemove == 0 and cmd.in_forward == 0 and cmd.in_back == 0 and cmd.in_moveleft == 0 and cmd.in_moveright == 0 then
        if distance >= 0.5 then
            local fwd1 = new_pos - lp_pos

            local pos1 = new_pos + fwd1:normalized( )*10

            local fwd = pos1 - lp_pos
            local pitch, yaw = fwd:angles()

            if yaw == nil then
                return
            end

            cmd.move_yaw = yaw
            cmd.in_speed = 0

            cmd.in_moveleft, cmd.in_moveright = 0, 0
            cmd.sidemove = 0


            if distance > 8 then
                cmd.forwardmove = 900000
            else
                local wishspeed = math.min( 450, math.max( 1.1+entity.get_prop( lp, "m_flDuckAmount" ) * 10, distance * 9 ) )
                local vel = vector( entity.get_prop( lp, "m_vecAbsVelocity" ) ):length2d( )
                if vel >= math.min( 250, wishspeed )+15 then
                    cmd.forwardmove = 0
                    cmd.in_forward = 0
                else
                    cmd.forwardmove = math.max( 6, vel >= math.min( 250, wishspeed ) and wishspeed * 0.9 or wishspeed )
                    cmd.in_forward = 1
                end
            end
        end
    end
end

local function handle_peek( cmd )
    local lp = entity.get_local_player( )
    local lp_pos = vector( client.eye_position( ) )

    cache.found_position.z = lp_pos.z
    
    move_to_pos( cmd, lp, lp_pos, cache.found_position )
end

local function handle_retreat( cmd )
    local lp = entity.get_local_player( )
    local lp_pos = vector( client.eye_position( ) )

    move_to_pos( cmd, lp, lp_pos, cache.autopeek_position )
end

local function is_doubletap_charged( )
    local m_nTickBase = entity.get_prop( entity.get_local_player( ), 'm_nTickBase')
    local client_latency = client.latency( )
    local shift = math.floor( m_nTickBase - globals.tickcount( ) - 3 - toticks( client_latency ) * .5 + .7 * ( client_latency * 10 ) )

    local wanted = -11

    return shift <= wanted
end

local debug = {
    state = 'disabled',
    step = 0,
    visual_step = 0,
}

local function set_state( new_state )
    debug.state = new_state
end

local e_visual_steps = {
    IDLE = 0,
    FINDING_TARGET = 1,
    SEARCHING_HITPOINTS = 2,
    CHECKING_HITPOINTS = 3,
    PEEKING = 4,
    RETREATING = 5,
    WAITING_FOR_SHOT = 6,
    NO_ENEMIES = 7,
    DT_NOT_CHARGED = 8,
}

local visual_texts = {
    [ e_visual_steps.IDLE ] = 'freezed',
    [ e_visual_steps.FINDING_TARGET ] = 'waiting',
    [ e_visual_steps.SEARCHING_HITPOINTS ] = 'waiting',
    [ e_visual_steps.CHECKING_HITPOINTS ] = 'ensuring hitpoints',
    [ e_visual_steps.PEEKING ] = 'peeking',
    [ e_visual_steps.RETREATING ] = 'waiting',
    [ e_visual_steps.WAITING_FOR_SHOT ] = 'waiting for shot',
    [ e_visual_steps.NO_ENEMIES ] = 'waiting',
    [ e_visual_steps.DT_NOT_CHARGED ] = '[!] dt not fully charged [!]',
}

local e_steps = {
    IDLE = 0,
    FINDING_TARGET = 1,
    SEARCHING_HITPOINTS = 2,
    CHECKING_HITPOINTS = 3,
    PEEKING = 4,
    RETREATING = 5,
}

local function set_step( step )
    debug.step = step
end

local function set_visual_step( step )
    debug.visual_step = step
end

local peeking_points = { }
local function gpt_peek( cmd )
    local lp = entity.get_local_player( )

    if not lp then return end

    local pos = vector( entity.get_prop( lp, 'm_vecOrigin' ) )
    local autopeek_state = ui.get( autopeek[ 2 ] )
    if not autopeek_state  or not lua_items.ragebot.ai_peek:get() then
        cache.autopeek_position = pos
        peeking_points = get_peeking_points( lp )

        reset_cache( )
        set_state( 'disabled' )
        set_step( e_steps.IDLE )
        set_visual_step( e_visual_steps.IDLE )
        return
    end

    set_visual_peeking_points( peeking_points )

    set_state( 'idle' )

    local distance = ( cache.autopeek_position - cache.found_position ):length2d( )
    local can_run = can_hit_in_x_ticks( distance, 250, 24 )

    local can_shoot = ready_to_shoot( lp, cmd )

    if ( cache.last_seen + 24 ) >= globals.tickcount( ) and can_run and can_shoot then
        handle_peek( cmd )
        set_state( 'peeking' )
        set_step( e_steps.PEEKING )
        set_visual_step( e_visual_steps.PEEKING )
        return --!!!!!!!!!!!!!!!!!!! REMOVE IF BAD
    elseif ( cache.autopeek_position - pos ):length2d( ) > 5 then
        handle_retreat( cmd )
        set_state( 'retreating' )
        
        set_step( e_steps.RETREATING )
        set_visual_step( e_visual_steps.RETREATING )
        return --!!!!!!!!!!!!!!!!!!! REMOVE IF BAD
    end

    if not can_shoot then
        set_step( e_steps.IDLE )
        set_visual_step( e_visual_steps.WAITING_FOR_SHOT )
        return
    end
    
    local targets = { closest_enemy }

    if #targets == 0 or targets[ 1 ] == nil then
        reset_cache( )
        set_state( 'idle' )
        set_step( e_steps.IDLE )
        set_visual_step( e_visual_steps.NO_ENEMIES )
        return
    end

    if can_shoot then
        set_step( e_steps.FINDING_TARGET )
        set_visual_step( e_visual_steps.FINDING_TARGET )
    end

    local g_can_hit = false
    local g_can_peek = false
    for idx = 1, #targets do
        local target = targets[ idx ]

        if not entity.is_alive( target ) or entity.is_dormant( target ) then
            goto continue
        end
        
        local target_data = enemy_lc_data[ target ]

        if target_data == nil then
            target_data = create_new_record( target )
        end

        target_data.update( )

        local is_lc = target_data.breaking_lc
        local is_defensive =  target_data.defensive
        local can_peek = not is_lc and not is_defensive

        local hitpoints = get_player_points( target )
        debug_visualize( hitpoints, target )

        set_step( e_steps.SEARCHING_HITPOINTS )
        set_visual_step( e_visual_steps.SEARCHING_HITPOINTS )


        if not can_peek then
            set_step( e_steps.CHECKING_HITPOINTS )
            set_visual_step( e_visual_steps.CHECKING_HITPOINTS )
            goto continue
        end

        local can_hit = can_hit_from_positions( lp, peeking_points, target, hitpoints )

        if can_hit then
            g_can_hit = true
            g_can_peek = true
            break
        end

        ::continue::
    end

    local dt_charged = is_doubletap_charged( )

    if not dt_charged then
        set_step( e_steps.FINDING_TARGET )
        set_visual_step( e_visual_steps.DT_NOT_CHARGED )
    end

    if g_can_hit and g_can_peek and dt_charged then
        cache.last_seen = globals.tickcount( )
    elseif not g_can_peek and not dt_charged then
        set_state( 'can\'t peek' )
    elseif not dt_charged then
        set_state( 'dt not charged' )
    end
end

local x, y = 400, 400

local visual_progressbar = {
    lerped_pos = vector( 0, 0, 0 ),
    gap = 30,
    radius = 5,
    pad = vector( 3, 2, 0 )
}

local func = {
    RGBAtoHEX = function(redArg, greenArg, blueArg, alphaArg)
        return string.format('%.2x%.2x%.2x%.2x', redArg, greenArg, blueArg, alphaArg)
    end,
}

local animate_text = function(time, string, r, g, b, a)
    local t_out, t_out_iter = { }, 1

    local l = string:len( ) - 1
    local mainClr = {}
    mainClr.r, mainClr.g, mainClr.b, mainClr.a = 255, 255, 255, 255


    local r_add = (mainClr.r - r)
    local g_add = (mainClr.g - g)
    local b_add = (mainClr.b - b)
    local a_add = (mainClr.a - a)

    for i = 1, #string do
        local iter = (i - 1)/(#string - 1) + time
        t_out[t_out_iter] = "\a" .. func.RGBAtoHEX( r + r_add * math.abs(math.cos( iter )), g + g_add * math.abs(math.cos( iter )), b + b_add * math.abs(math.cos( iter )), a + a_add * math.abs(math.cos( iter )) )

        t_out[t_out_iter + 1] = string:sub( i, i )

        t_out_iter = t_out_iter + 2
    end

    return t_out
end


local function render_screen_bar( )
    local lp = entity.get_local_player( )
    if not lp then return end

    for target, tbl in pairs( debug_visuals ) do
        if entity.is_alive( target ) and not entity.is_dormant( target ) then
            for i = 1, #tbl do
                local position = tbl[ i ]
        
                local s_x, s_y = renderer.world_to_screen( position.x, position.y, position.z )
        
                if s_x ~= nil and s_y ~= nil then
                    -- renderer.circle(x, y, r, g, b, a, radius, start_degrees, percentage)
                    renderer.circle( s_x, s_y, 255, 255, 255, 150, 2, 0, 1 )
                end
            end
        end
    end

    local step = debug.step
    if step == e_steps.RETREATING then
        set_step( e_steps.IDLE )
        step = debug.step
    end

    local screen = { client.screen_size( ) }
    screen = vector( screen[ 1 ] / 2, screen[ 2 ] - 100, 0 )

    local null_pos = vector( screen.x - ( 2 * visual_progressbar.gap ), screen.y, 0 )
    local active_pos = vector( screen.x + ( ( step - 2 ) * visual_progressbar.gap ), screen.y, 0 )

    if visual_progressbar.lerped_pos.x == 0 then
        visual_progressbar.lerped_pos = active_pos
    end

    local max_width = 5 * visual_progressbar.gap
    visual_progressbar.lerped_pos = visual_progressbar.lerped_pos:lerp( active_pos, 0.1 )
    visual_progressbar.x = visual_progressbar.lerped_pos.x + visual_progressbar.radius * 2
    local progress = ( visual_progressbar.lerped_pos.x - null_pos.x ) / max_width

    for i = -2, 2, 1 do
        local pos = vector( screen.x + ( i * visual_progressbar.gap ), screen.y, 0 )
    end



    local lp_pos = vector( entity.get_prop( lp, 'm_vecOrigin' ) )
    local dist_to_point = ( lp_pos - cache.found_position ):length2d( )
    local progress_to_point = ( cache.found_position_dist - dist_to_point ) / cache.found_position_dist

    if progress_to_point < 0 then
        progress_to_point = 0
    end

    if progress_to_point > 1 then
        progress_to_point = 1
    end

    for i = -2, 2, 1 do
        local draw_step = i + 2
        local pos = vector( screen.x + ( i * visual_progressbar.gap ), screen.y, 0 )

        local c = ( visual_progressbar.lerped_pos.x + visual_progressbar.radius ) >= pos.x and { 0, 255, 0 } or { 0, 0, 0 }
        local mainClr = {}
        mainClr.r, mainClr.g, mainClr.b, mainClr.a = 55,55,55,255
        local w,h = client.screen_size()
        local text = animate_text(globals.curtime(), "cd+" and visual_texts[ debug.visual_step ]:lower(), mainClr.r, mainClr.g, mainClr.b, 255)

        if draw_step == debug.step then
            renderer.text(
                w / 2, visual_progressbar.lerped_pos.y / 2,
                255, 255, 255, 255,
                'cd', 0,
                unpack(text)
            )
        end
    end
end

local visual_points = {
    last_pressed = 0,
    last_state = false,
    animation_time = .2,
}

local function ease_in_back( time )
    local c1 = 1.70158
    local c3 = c1 + 1

    return c3 * time * time * time - c1 * time * time
end

local function render_peeking_point( pos, state )
    renderer.circle(
        pos.x, pos.y,
        255, 255, 255, 255,
        2, 0, 1
    )

    if state then
        renderer.circle_outline(
            pos.x, pos.y,
            0, 255, 0, 255,
            3, 0, 1,
            2
        )
    end
end

local function render_peeking_points( )
    local lp = entity.get_local_player( )
    local preview =  lp and ui.is_menu_open()
    if preview then
        visuals.peeking_points = get_peeking_points( lp )
    end

    local ap_state = ui.get( autopeek[ 2 ] )

    if ap_state ~= visual_points.last_state then
        visual_points.last_pressed = globals.curtime( )
        visual_points.last_state = ap_state
    end

    local diff = globals.curtime( ) - visual_points.last_pressed
    diff = math.min( diff, visual_points.animation_time )

    local animation_factor = ease_in_back( diff / visual_points.animation_time )

    if not ap_state then
        animation_factor = 1 - animation_factor
    end

    if preview then
        animation_factor = 1
    end

    if not ap_state and animation_factor <= 0.1 then return end

    local points = visuals.peeking_points

    local lp_pos = vector( entity.get_origin( lp ) )

    for i = 1, #points do
        local point = points[ i ]

        local pos = vector( point.x, point.y, point.z )
        local pos_diff = pos - lp_pos

        pos_diff.x = pos_diff.x * animation_factor
        pos_diff.y = pos_diff.y * animation_factor

        pos = lp_pos + pos_diff

        pos = vector( renderer.world_to_screen( pos.x, pos.y, pos.z ) )

        if pos.x ~= nil and pos.y ~= nil then
            render_peeking_point( pos, visuals.found_point == i )
        end
    end
end


local config_cfg = {lua_items, antiaim_builder}

local package, data, encrypted, decrypted = pui.setup(config_cfg), "", "", ""
config = {}

local cfg_system = {}
configs_db = database.read(lua_db.config) or { }
configs_db.cfg_list = configs_db.cfg_list or {{'Default', 'W3sibWFpbiI6eyJ0YWIiOiJDb25maWcifSwiY29uZmlnIjp7Imxpc3QiOjAsIm5hbWUiOiIifSwiYW50aWFpbSI6eyJ0YWIiOiJCdWlsZGVyIiwiZnJlZXNfcGVlayI6dHJ1ZSwiYWRkb25zIjpbIlNhZmUgSGVhZCIsIn4iXSwia2V5X2ZvcndhcmQiOlsxLDAsIn4iXSwia2V5X2xlZnQiOlsxLDkwLCJ+Il0sInNhZmVfaGVhZCI6WyJ+Il0sIm1hbnVhbF9wZWVrIjp0cnVlLCJ5YXdfZGlyZWN0aW9uIjpbIkZyZWVzdGFuZGluZyIsIk1hbnVhbCIsIn4iXSwieWF3X2Jhc2UiOiJBdCB0YXJnZXRzIiwia2V5X2ZyZWVzdGFuZCI6WzEsNiwifiJdLCJrZXlfcmlnaHQiOlsxLDY3LCJ+Il0sImNvbmRpdGlvbiI6Ilx1MDAwYlN0YW5kXHIifSwibWlzYyI6eyJjcm9zc19jb2xvciI6ZmFsc2UsImxvZyI6ZmFsc2UsImNyb3NzX2luZF9jIjoiI0ZGRkZGRkZGIiwiY3Jvc3NfY29sb3JfYyI6IiM2NDY0RkZGRiIsImFuaW1hdGlvbiI6ZmFsc2UsImluZm9fcGFuZWwiOmZhbHNlLCJjcm9zc19pbmRfdHlwZSI6Ik5ld2VzdCIsImtleV9jb2xvcl9jIjoiI0ZGRkZGRkZGIiwidGVsZXBvcnRfa2V5IjpbMSwwLCJ+Il0sImtleV9jb2xvciI6ZmFsc2UsInZlbG9jaXR5X3dpbmRvdyI6ZmFsc2UsImRlZmVuc2l2ZV9zdHlsZSI6IkRlZmF1bHQiLCJkZWZlbnNpdmVfd2luZG93IjpmYWxzZSwidGVsZXBvcnQiOmZhbHNlLCJhc3BlY3RyYXRpbyI6dHJ1ZSwiYW5pbWF0aW9uX2FpciI6Ik9mZiIsImZhc3RfbGFkZGVyIjp0cnVlLCJ2ZWxvY2l0eV93aW5kb3dfYyI6IiNGRkZGRkZGRiIsInJlc29sdmVyIjpmYWxzZSwic3BhbW1lcnMiOlsifiJdLCJjcm9zc19pbmQiOmZhbHNlLCJ0aGlyZF9wZXJzb25fdmFsdWUiOjQ0LCJhc3BlY3RyYXRpb192YWx1ZSI6MTMzLCJ0aGlyZF9wZXJzb24iOnRydWUsImFuaW1hdGlvbl92YWx1ZSI6NSwidmVsb2NpdHlfc3R5bGUiOiJEZWZhdWx0IiwiZGVmZW5zaXZlX3dpbmRvd19jIjoiI0ZGRkZGRkZGIiwiYW5pbWF0aW9uX2dyb3VuZCI6IlN0YXRpYyJ9fSxbeyJlbmFibGUiOmZhbHNlLCJ5YXdfdHlwZSI6IkRlZmF1bHQiLCJtb2RfdHlwZSI6IkNlbnRlciIsImRlZmVuc2l2ZV9waXRjaCI6IkN1c3RvbSIsImJvZHlfc2xpZGVyIjoxLCJtb2RfdGlja3MiOjAsInBlZWtfZGVmIjpmYWxzZSwiZGVmZW5zaXZlIjpmYWxzZSwiZGVmX2JvZHlfeWF3X3R5cGUiOiJPZmYiLCJmb3JjZV9kZWYiOnRydWUsInlhd19kZWxheSI6MSwicGl0Y2hfdmFsdWUiOi04OSwiZGVmX2JvZHlfc2xpZGVyIjowLCJkZWZfbW9kX3R5cGUiOiJPZmYiLCJ5YXdfdmFsdWUiOjAsInlhd19yYW5kb20iOjIwLCJkZWZlbnNpdmVfeWF3IjoiT2ZmIiwiYm9keV95YXdfdHlwZSI6IkppdHRlciIsInlhd19yaWdodCI6MzAsIm1vZF9kbSI6MTcsInlhd19sZWZ0IjotMjgsImRlZmVuc2l2ZV90eXBlIjoiRGVmYXVsdCIsImRlZl9tb2RfZG0iOjB9LHsiZW5hYmxlIjpmYWxzZSwieWF3X3R5cGUiOiJEZWZhdWx0IiwibW9kX3R5cGUiOiJDZW50ZXIiLCJkZWZlbnNpdmVfcGl0Y2giOiJDdXN0b20iLCJib2R5X3NsaWRlciI6MSwibW9kX3RpY2tzIjowLCJwZWVrX2RlZiI6ZmFsc2UsImRlZmVuc2l2ZSI6ZmFsc2UsImRlZl9ib2R5X3lhd190eXBlIjoiT2ZmIiwiZm9yY2VfZGVmIjp0cnVlLCJ5YXdfZGVsYXkiOjEsInBpdGNoX3ZhbHVlIjotODksImRlZl9ib2R5X3NsaWRlciI6MCwiZGVmX21vZF90eXBlIjoiT2ZmIiwieWF3X3ZhbHVlIjowLCJ5YXdfcmFuZG9tIjoyMCwiZGVmZW5zaXZlX3lhdyI6Ik9mZiIsImJvZHlfeWF3X3R5cGUiOiJKaXR0ZXIiLCJ5YXdfcmlnaHQiOjMwLCJtb2RfZG0iOjE3LCJ5YXdfbGVmdCI6LTI4LCJkZWZlbnNpdmVfdHlwZSI6IkRlZmF1bHQiLCJkZWZfbW9kX2RtIjowfSx7ImVuYWJsZSI6ZmFsc2UsInlhd190eXBlIjoiRGVmYXVsdCIsIm1vZF90eXBlIjoiQ2VudGVyIiwiZGVmZW5zaXZlX3BpdGNoIjoiQ3VzdG9tIiwiYm9keV9zbGlkZXIiOjEsIm1vZF90aWNrcyI6MCwicGVla19kZWYiOmZhbHNlLCJkZWZlbnNpdmUiOmZhbHNlLCJkZWZfYm9keV95YXdfdHlwZSI6Ik9mZiIsImZvcmNlX2RlZiI6dHJ1ZSwieWF3X2RlbGF5IjoxLCJwaXRjaF92YWx1ZSI6LTg5LCJkZWZfYm9keV9zbGlkZXIiOjAsImRlZl9tb2RfdHlwZSI6Ik9mZiIsInlhd192YWx1ZSI6MCwieWF3X3JhbmRvbSI6MjAsImRlZmVuc2l2ZV95YXciOiJPZmYiLCJib2R5X3lhd190eXBlIjoiSml0dGVyIiwieWF3X3JpZ2h0IjozMCwibW9kX2RtIjoxNywieWF3X2xlZnQiOi0yOCwiZGVmZW5zaXZlX3R5cGUiOiJEZWZhdWx0IiwiZGVmX21vZF9kbSI6MH0seyJlbmFibGUiOmZhbHNlLCJ5YXdfdHlwZSI6IkRlZmF1bHQiLCJtb2RfdHlwZSI6Ik9mZiIsImRlZmVuc2l2ZV9waXRjaCI6Ik9mZiIsImJvZHlfc2xpZGVyIjowLCJtb2RfdGlja3MiOjAsInBlZWtfZGVmIjpmYWxzZSwiZGVmZW5zaXZlIjpmYWxzZSwiZGVmX2JvZHlfeWF3X3R5cGUiOiJPZmYiLCJmb3JjZV9kZWYiOmZhbHNlLCJ5YXdfZGVsYXkiOjQsInBpdGNoX3ZhbHVlIjowLCJkZWZfYm9keV9zbGlkZXIiOjAsImRlZl9tb2RfdHlwZSI6Ik9mZiIsInlhd192YWx1ZSI6MCwieWF3X3JhbmRvbSI6MCwiZGVmZW5zaXZlX3lhdyI6Ik9mZiIsImJvZHlfeWF3X3R5cGUiOiJPZmYiLCJ5YXdfcmlnaHQiOjAsIm1vZF9kbSI6MCwieWF3X2xlZnQiOjAsImRlZmVuc2l2ZV90eXBlIjoiRGVmYXVsdCIsImRlZl9tb2RfZG0iOjB9LHsiZW5hYmxlIjp0cnVlLCJ5YXdfdHlwZSI6IkRlZmF1bHQiLCJtb2RfdHlwZSI6Ik9mZiIsImRlZmVuc2l2ZV9waXRjaCI6Ik9mZiIsImJvZHlfc2xpZGVyIjotMSwibW9kX3RpY2tzIjowLCJwZWVrX2RlZiI6ZmFsc2UsImRlZmVuc2l2ZSI6dHJ1ZSwiZGVmX2JvZHlfeWF3X3R5cGUiOiJPZmYiLCJmb3JjZV9kZWYiOnRydWUsInlhd19kZWxheSI6MSwicGl0Y2hfdmFsdWUiOjAsImRlZl9ib2R5X3NsaWRlciI6MCwiZGVmX21vZF90eXBlIjoiT2ZmIiwieWF3X3ZhbHVlIjotMjMsInlhd19yYW5kb20iOjEwLCJkZWZlbnNpdmVfeWF3IjoiU3BpbiIsImJvZHlfeWF3X3R5cGUiOiJKaXR0ZXIiLCJ5YXdfcmlnaHQiOjMxLCJtb2RfZG0iOjAsInlhd19sZWZ0IjotMTksImRlZmVuc2l2ZV90eXBlIjoiRGVmYXVsdCIsImRlZl9tb2RfZG0iOjB9LHsiZW5hYmxlIjp0cnVlLCJ5YXdfdHlwZSI6IkRlZmF1bHQiLCJtb2RfdHlwZSI6Ik9mZiIsImRlZmVuc2l2ZV9waXRjaCI6IkN1c3RvbSIsImJvZHlfc2xpZGVyIjotMSwibW9kX3RpY2tzIjowLCJwZWVrX2RlZiI6ZmFsc2UsImRlZmVuc2l2ZSI6dHJ1ZSwiZGVmX2JvZHlfeWF3X3R5cGUiOiJPZmYiLCJmb3JjZV9kZWYiOnRydWUsInlhd19kZWxheSI6MSwicGl0Y2hfdmFsdWUiOjg5LCJkZWZfYm9keV9zbGlkZXIiOjAsImRlZl9tb2RfdHlwZSI6Ik9mZiIsInlhd192YWx1ZSI6LTM2LCJ5YXdfcmFuZG9tIjoxMCwiZGVmZW5zaXZlX3lhdyI6IlNwaW4iLCJib2R5X3lhd190eXBlIjoiSml0dGVyIiwieWF3X3JpZ2h0IjozMCwibW9kX2RtIjoxNCwieWF3X2xlZnQiOi0yMiwiZGVmZW5zaXZlX3R5cGUiOiJEZWZhdWx0IiwiZGVmX21vZF9kbSI6MH0seyJlbmFibGUiOnRydWUsInlhd190eXBlIjoiRGVmYXVsdCIsIm1vZF90eXBlIjoiT2ZmIiwiZGVmZW5zaXZlX3BpdGNoIjoiQ3VzdG9tIiwiYm9keV9zbGlkZXIiOi0xLCJtb2RfdGlja3MiOjAsInBlZWtfZGVmIjpmYWxzZSwiZGVmZW5zaXZlIjp0cnVlLCJkZWZfYm9keV95YXdfdHlwZSI6Ik9mZiIsImZvcmNlX2RlZiI6dHJ1ZSwieWF3X2RlbGF5IjoxLCJwaXRjaF92YWx1ZSI6LTg5LCJkZWZfYm9keV9zbGlkZXIiOjAsImRlZl9tb2RfdHlwZSI6Ik9mZiIsInlhd192YWx1ZSI6LTE4LCJ5YXdfcmFuZG9tIjoxMCwiZGVmZW5zaXZlX3lhdyI6IlNwaW4iLCJib2R5X3lhd190eXBlIjoiSml0dGVyIiwieWF3X3JpZ2h0IjoyNCwibW9kX2RtIjowLCJ5YXdfbGVmdCI6LTIwLCJkZWZlbnNpdmVfdHlwZSI6IkRlZmF1bHQiLCJkZWZfbW9kX2RtIjowfSx7ImVuYWJsZSI6dHJ1ZSwieWF3X3R5cGUiOiJEZWZhdWx0IiwibW9kX3R5cGUiOiJDZW50ZXIiLCJkZWZlbnNpdmVfcGl0Y2giOiJDdXN0b20iLCJib2R5X3NsaWRlciI6MSwibW9kX3RpY2tzIjowLCJwZWVrX2RlZiI6ZmFsc2UsImRlZmVuc2l2ZSI6dHJ1ZSwiZGVmX2JvZHlfeWF3X3R5cGUiOiJPZmYiLCJmb3JjZV9kZWYiOnRydWUsInlhd19kZWxheSI6MSwicGl0Y2hfdmFsdWUiOi04OSwiZGVmX2JvZHlfc2xpZGVyIjowLCJkZWZfbW9kX3R5cGUiOiJPZmYiLCJ5YXdfdmFsdWUiOi0zOSwieWF3X3JhbmRvbSI6MjAsImRlZmVuc2l2ZV95YXciOiJTcGluIiwiYm9keV95YXdfdHlwZSI6IkppdHRlciIsInlhd19yaWdodCI6MzAsIm1vZF9kbSI6MTcsInlhd19sZWZ0IjotMTgsImRlZmVuc2l2ZV90eXBlIjoiRGVmYXVsdCIsImRlZl9tb2RfZG0iOjB9XV0='}}
configs_db.menu_list = configs_db.menu_list or {'Default'}

configs_db.cfg_list[1][2] = "W3sibWFpbiI6eyJ0YWIiOiJDb25maWcifSwiY29uZmlnIjp7Imxpc3QiOjAsIm5hbWUiOiIifSwiYW50aWFpbSI6eyJ0YWIiOiJCdWlsZGVyIiwiZnJlZXNfcGVlayI6dHJ1ZSwiYWRkb25zIjpbIlNhZmUgSGVhZCIsIn4iXSwia2V5X2ZvcndhcmQiOlsxLDAsIn4iXSwia2V5X2xlZnQiOlsxLDkwLCJ+Il0sInNhZmVfaGVhZCI6WyJ+Il0sIm1hbnVhbF9wZWVrIjp0cnVlLCJ5YXdfZGlyZWN0aW9uIjpbIkZyZWVzdGFuZGluZyIsIk1hbnVhbCIsIn4iXSwieWF3X2Jhc2UiOiJBdCB0YXJnZXRzIiwia2V5X2ZyZWVzdGFuZCI6WzEsNiwifiJdLCJrZXlfcmlnaHQiOlsxLDY3LCJ+Il0sImNvbmRpdGlvbiI6Ilx1MDAwYlN0YW5kXHIifSwibWlzYyI6eyJjcm9zc19jb2xvciI6ZmFsc2UsImxvZyI6ZmFsc2UsImNyb3NzX2luZF9jIjoiI0ZGRkZGRkZGIiwiY3Jvc3NfY29sb3JfYyI6IiM2NDY0RkZGRiIsImFuaW1hdGlvbiI6ZmFsc2UsImluZm9fcGFuZWwiOmZhbHNlLCJjcm9zc19pbmRfdHlwZSI6Ik5ld2VzdCIsImtleV9jb2xvcl9jIjoiI0ZGRkZGRkZGIiwidGVsZXBvcnRfa2V5IjpbMSwwLCJ+Il0sImtleV9jb2xvciI6ZmFsc2UsInZlbG9jaXR5X3dpbmRvdyI6ZmFsc2UsImRlZmVuc2l2ZV9zdHlsZSI6IkRlZmF1bHQiLCJkZWZlbnNpdmVfd2luZG93IjpmYWxzZSwidGVsZXBvcnQiOmZhbHNlLCJhc3BlY3RyYXRpbyI6dHJ1ZSwiYW5pbWF0aW9uX2FpciI6Ik9mZiIsImZhc3RfbGFkZGVyIjp0cnVlLCJ2ZWxvY2l0eV93aW5kb3dfYyI6IiNGRkZGRkZGRiIsInJlc29sdmVyIjpmYWxzZSwic3BhbW1lcnMiOlsifiJdLCJjcm9zc19pbmQiOmZhbHNlLCJ0aGlyZF9wZXJzb25fdmFsdWUiOjQ0LCJhc3BlY3RyYXRpb192YWx1ZSI6MTMzLCJ0aGlyZF9wZXJzb24iOnRydWUsImFuaW1hdGlvbl92YWx1ZSI6NSwidmVsb2NpdHlfc3R5bGUiOiJEZWZhdWx0IiwiZGVmZW5zaXZlX3dpbmRvd19jIjoiI0ZGRkZGRkZGIiwiYW5pbWF0aW9uX2dyb3VuZCI6IlN0YXRpYyJ9fSxbeyJlbmFibGUiOmZhbHNlLCJ5YXdfdHlwZSI6IkRlZmF1bHQiLCJtb2RfdHlwZSI6IkNlbnRlciIsImRlZmVuc2l2ZV9waXRjaCI6IkN1c3RvbSIsImJvZHlfc2xpZGVyIjoxLCJtb2RfdGlja3MiOjAsInBlZWtfZGVmIjpmYWxzZSwiZGVmZW5zaXZlIjpmYWxzZSwiZGVmX2JvZHlfeWF3X3R5cGUiOiJPZmYiLCJmb3JjZV9kZWYiOnRydWUsInlhd19kZWxheSI6MSwicGl0Y2hfdmFsdWUiOi04OSwiZGVmX2JvZHlfc2xpZGVyIjowLCJkZWZfbW9kX3R5cGUiOiJPZmYiLCJ5YXdfdmFsdWUiOjAsInlhd19yYW5kb20iOjIwLCJkZWZlbnNpdmVfeWF3IjoiT2ZmIiwiYm9keV95YXdfdHlwZSI6IkppdHRlciIsInlhd19yaWdodCI6MzAsIm1vZF9kbSI6MTcsInlhd19sZWZ0IjotMjgsImRlZmVuc2l2ZV90eXBlIjoiRGVmYXVsdCIsImRlZl9tb2RfZG0iOjB9LHsiZW5hYmxlIjpmYWxzZSwieWF3X3R5cGUiOiJEZWZhdWx0IiwibW9kX3R5cGUiOiJDZW50ZXIiLCJkZWZlbnNpdmVfcGl0Y2giOiJDdXN0b20iLCJib2R5X3NsaWRlciI6MSwibW9kX3RpY2tzIjowLCJwZWVrX2RlZiI6ZmFsc2UsImRlZmVuc2l2ZSI6ZmFsc2UsImRlZl9ib2R5X3lhd190eXBlIjoiT2ZmIiwiZm9yY2VfZGVmIjp0cnVlLCJ5YXdfZGVsYXkiOjEsInBpdGNoX3ZhbHVlIjotODksImRlZl9ib2R5X3NsaWRlciI6MCwiZGVmX21vZF90eXBlIjoiT2ZmIiwieWF3X3ZhbHVlIjowLCJ5YXdfcmFuZG9tIjoyMCwiZGVmZW5zaXZlX3lhdyI6Ik9mZiIsImJvZHlfeWF3X3R5cGUiOiJKaXR0ZXIiLCJ5YXdfcmlnaHQiOjMwLCJtb2RfZG0iOjE3LCJ5YXdfbGVmdCI6LTI4LCJkZWZlbnNpdmVfdHlwZSI6IkRlZmF1bHQiLCJkZWZfbW9kX2RtIjowfSx7ImVuYWJsZSI6ZmFsc2UsInlhd190eXBlIjoiRGVmYXVsdCIsIm1vZF90eXBlIjoiQ2VudGVyIiwiZGVmZW5zaXZlX3BpdGNoIjoiQ3VzdG9tIiwiYm9keV9zbGlkZXIiOjEsIm1vZF90aWNrcyI6MCwicGVla19kZWYiOmZhbHNlLCJkZWZlbnNpdmUiOmZhbHNlLCJkZWZfYm9keV95YXdfdHlwZSI6Ik9mZiIsImZvcmNlX2RlZiI6dHJ1ZSwieWF3X2RlbGF5IjoxLCJwaXRjaF92YWx1ZSI6LTg5LCJkZWZfYm9keV9zbGlkZXIiOjAsImRlZl9tb2RfdHlwZSI6Ik9mZiIsInlhd192YWx1ZSI6MCwieWF3X3JhbmRvbSI6MjAsImRlZmVuc2l2ZV95YXciOiJPZmYiLCJib2R5X3lhd190eXBlIjoiSml0dGVyIiwieWF3X3JpZ2h0IjozMCwibW9kX2RtIjoxNywieWF3X2xlZnQiOi0yOCwiZGVmZW5zaXZlX3R5cGUiOiJEZWZhdWx0IiwiZGVmX21vZF9kbSI6MH0seyJlbmFibGUiOmZhbHNlLCJ5YXdfdHlwZSI6IkRlZmF1bHQiLCJtb2RfdHlwZSI6Ik9mZiIsImRlZmVuc2l2ZV9waXRjaCI6Ik9mZiIsImJvZHlfc2xpZGVyIjowLCJtb2RfdGlja3MiOjAsInBlZWtfZGVmIjpmYWxzZSwiZGVmZW5zaXZlIjpmYWxzZSwiZGVmX2JvZHlfeWF3X3R5cGUiOiJPZmYiLCJmb3JjZV9kZWYiOmZhbHNlLCJ5YXdfZGVsYXkiOjQsInBpdGNoX3ZhbHVlIjowLCJkZWZfYm9keV9zbGlkZXIiOjAsImRlZl9tb2RfdHlwZSI6Ik9mZiIsInlhd192YWx1ZSI6MCwieWF3X3JhbmRvbSI6MCwiZGVmZW5zaXZlX3lhdyI6Ik9mZiIsImJvZHlfeWF3X3R5cGUiOiJPZmYiLCJ5YXdfcmlnaHQiOjAsIm1vZF9kbSI6MCwieWF3X2xlZnQiOjAsImRlZmVuc2l2ZV90eXBlIjoiRGVmYXVsdCIsImRlZl9tb2RfZG0iOjB9LHsiZW5hYmxlIjp0cnVlLCJ5YXdfdHlwZSI6IkRlZmF1bHQiLCJtb2RfdHlwZSI6Ik9mZiIsImRlZmVuc2l2ZV9waXRjaCI6Ik9mZiIsImJvZHlfc2xpZGVyIjotMSwibW9kX3RpY2tzIjowLCJwZWVrX2RlZiI6ZmFsc2UsImRlZmVuc2l2ZSI6dHJ1ZSwiZGVmX2JvZHlfeWF3X3R5cGUiOiJPZmYiLCJmb3JjZV9kZWYiOnRydWUsInlhd19kZWxheSI6MSwicGl0Y2hfdmFsdWUiOjAsImRlZl9ib2R5X3NsaWRlciI6MCwiZGVmX21vZF90eXBlIjoiT2ZmIiwieWF3X3ZhbHVlIjotMjMsInlhd19yYW5kb20iOjEwLCJkZWZlbnNpdmVfeWF3IjoiU3BpbiIsImJvZHlfeWF3X3R5cGUiOiJKaXR0ZXIiLCJ5YXdfcmlnaHQiOjMxLCJtb2RfZG0iOjAsInlhd19sZWZ0IjotMTksImRlZmVuc2l2ZV90eXBlIjoiRGVmYXVsdCIsImRlZl9tb2RfZG0iOjB9LHsiZW5hYmxlIjp0cnVlLCJ5YXdfdHlwZSI6IkRlZmF1bHQiLCJtb2RfdHlwZSI6Ik9mZiIsImRlZmVuc2l2ZV9waXRjaCI6IkN1c3RvbSIsImJvZHlfc2xpZGVyIjotMSwibW9kX3RpY2tzIjowLCJwZWVrX2RlZiI6ZmFsc2UsImRlZmVuc2l2ZSI6dHJ1ZSwiZGVmX2JvZHlfeWF3X3R5cGUiOiJPZmYiLCJmb3JjZV9kZWYiOnRydWUsInlhd19kZWxheSI6MSwicGl0Y2hfdmFsdWUiOjg5LCJkZWZfYm9keV9zbGlkZXIiOjAsImRlZl9tb2RfdHlwZSI6Ik9mZiIsInlhd192YWx1ZSI6LTM2LCJ5YXdfcmFuZG9tIjoxMCwiZGVmZW5zaXZlX3lhdyI6IlNwaW4iLCJib2R5X3lhd190eXBlIjoiSml0dGVyIiwieWF3X3JpZ2h0IjozMCwibW9kX2RtIjoxNCwieWF3X2xlZnQiOi0yMiwiZGVmZW5zaXZlX3R5cGUiOiJEZWZhdWx0IiwiZGVmX21vZF9kbSI6MH0seyJlbmFibGUiOnRydWUsInlhd190eXBlIjoiRGVmYXVsdCIsIm1vZF90eXBlIjoiT2ZmIiwiZGVmZW5zaXZlX3BpdGNoIjoiQ3VzdG9tIiwiYm9keV9zbGlkZXIiOi0xLCJtb2RfdGlja3MiOjAsInBlZWtfZGVmIjpmYWxzZSwiZGVmZW5zaXZlIjp0cnVlLCJkZWZfYm9keV95YXdfdHlwZSI6Ik9mZiIsImZvcmNlX2RlZiI6dHJ1ZSwieWF3X2RlbGF5IjoxLCJwaXRjaF92YWx1ZSI6LTg5LCJkZWZfYm9keV9zbGlkZXIiOjAsImRlZl9tb2RfdHlwZSI6Ik9mZiIsInlhd192YWx1ZSI6LTE4LCJ5YXdfcmFuZG9tIjoxMCwiZGVmZW5zaXZlX3lhdyI6IlNwaW4iLCJib2R5X3lhd190eXBlIjoiSml0dGVyIiwieWF3X3JpZ2h0IjoyNCwibW9kX2RtIjowLCJ5YXdfbGVmdCI6LTIwLCJkZWZlbnNpdmVfdHlwZSI6IkRlZmF1bHQiLCJkZWZfbW9kX2RtIjowfSx7ImVuYWJsZSI6dHJ1ZSwieWF3X3R5cGUiOiJEZWZhdWx0IiwibW9kX3R5cGUiOiJDZW50ZXIiLCJkZWZlbnNpdmVfcGl0Y2giOiJDdXN0b20iLCJib2R5X3NsaWRlciI6MSwibW9kX3RpY2tzIjowLCJwZWVrX2RlZiI6ZmFsc2UsImRlZmVuc2l2ZSI6dHJ1ZSwiZGVmX2JvZHlfeWF3X3R5cGUiOiJPZmYiLCJmb3JjZV9kZWYiOnRydWUsInlhd19kZWxheSI6MSwicGl0Y2hfdmFsdWUiOi04OSwiZGVmX2JvZHlfc2xpZGVyIjowLCJkZWZfbW9kX3R5cGUiOiJPZmYiLCJ5YXdfdmFsdWUiOi0zOSwieWF3X3JhbmRvbSI6MjAsImRlZmVuc2l2ZV95YXciOiJTcGluIiwiYm9keV95YXdfdHlwZSI6IkppdHRlciIsInlhd19yaWdodCI6MzAsIm1vZF9kbSI6MTcsInlhd19sZWZ0IjotMTgsImRlZmVuc2l2ZV90eXBlIjoiRGVmYXVsdCIsImRlZl9tb2RfZG0iOjB9XV0="

cfg_system.save_config = function(id)
    if id == 1 then return end
    if configs_db.cfg_list[id] == nil then
        print("Error: config with id "..id.." does not exist.")
        return
    end

    if configs_db.cfg_list[id][2] == nil then
        print("Error: second part of config with id "..id.." does not exist.")
        return
    end
    
    local raw = package:save()
    configs_db.cfg_list[id][2] = base64.encode(json.stringify(raw))
    database.write(lua_db.config, configs_db)
    database.flush()
end

cfg_system.create_config = function(name)
    if type(name) ~= 'string' then return end

    if name == nil or name == '' or name == ' ' then
        print('Wrong Name')
        return
    end

    for i= #configs_db.menu_list, 1, -1 do
        if configs_db.menu_list[i] == name then
            print('Another config has the same name')
            return
        end
    end

    if #configs_db.cfg_list > 7 then
        print('Maximum number of configs: 8')
        return
    end

    local completed = {name, ''}
    table.insert(configs_db.cfg_list, completed)
    table.insert(configs_db.menu_list, name)
    database.write(lua_db.config, configs_db)
    database.flush()
end

cfg_system.remove_config = function(id)
    if id == 1 then return end
    local item = configs_db.cfg_list[id][1]

    for i= #configs_db.cfg_list, 1, -1 do
        if configs_db.cfg_list[i][1] == item then
            table.remove(configs_db.cfg_list, i)
            table.remove(configs_db.menu_list, i)
        end
    end

    database.write(lua_db.config, configs_db)
    database.flush()
end

cfg_system.load_config = function(id)
    if configs_db.cfg_list[id][2] == nil or configs_db.cfg_list[id][2] == '' then
        print("Problem with: "..id.." Config")
        return
    end

    if id > #configs_db.cfg_list then
        print("Problem with: "..id.." Config")
        return
    end

    package:load(json.parse(base64.decode(configs_db.cfg_list[id][2])))
end

lua_items.config.create:set_callback(function() 
    cfg_system.create_config(lua_items.config.name:get())
    lua_items.config.list:update(configs_db.menu_list)
end)

lua_items.config.load:set_callback(function() 
    cfg_system.load_config(lua_items.config.list:get() + 1)
    lua_items.config.list:update(configs_db.menu_list)
end)

lua_items.config.accept:set_callback(function() 
    cfg_system.save_config(lua_items.config.list:get() + 1)
end)

lua_items.config.accept2:set_callback(function() 
    cfg_system.remove_config(lua_items.config.list:get() + 1)
    lua_items.config.list:update(configs_db.menu_list)
end)

lua_items.config.import:set_callback(function() 
    local clean = string.match(clipboard.get(), "[%w%+%/]+%=*")
    package:load(json.parse(base64.decode(clean)))
end)

lua_items.config.export:set_callback(function() 
    clipboard.set(base64.encode(json.stringify(package:save())))
end)
lua_items.config.list:update(configs_db.menu_list)

client.set_event_callback("setup_command", function(cmd)
    aa_helpers.setup(cmd)

    if lua_items.misc.duck_on_fd:get() then
        misc_helpers.allow_duck(cmd)
    elseif fd_reset then
        override.unset(ref.fakeduck)
        fd_reset = false
    end

    if lua_items.ragebot.unsafe_charge:get() then
        misc_helpers.unsafe_charge(cmd)
    elseif charge_state then
        ui.set(ref.aimbot, true)
    end

    if lua_items.misc.fast_ladder:get() then
        misc_helpers.fastladder(cmd)
    end

    if lua_items.misc.fps_boost:get() and not lua_items.misc.fps_always:get() then
        misc_helpers.fps_boost(cmd.quick_stop)
    end

    if lua_items.ragebot.defensive_fix:get() then
        rage_helpers.defensive_fix(cmd)
    end

    if lua_items.ragebot.scout_airstop:get() and lua_items.ragebot.scout_airstop_bind:get() then
        rage_helpers.air_qs(cmd)
    end

    if lua_items.misc.cs2_view:get() then
        misc_helpers.cs2_view()
    end

    if lua_items.antiaim.edgeyaw_on_fd:get() then
        if ui.get(ref.fakeduck) then
            ui.set(ref.edgeyaw, true)
        else
            ui.set(ref.edgeyaw, false)
        end
    else
        ui.set(ref.edgeyaw, false)
    end

    if lua_items.misc.auto_tp:get() and lua_items.misc.auto_tp_key:get() then
        misc_helpers.auto_tp(cmd)
    end

    if lua_items.misc.airlag:get() then
        misc_helpers.airlag(cmd)
    end

    if ui.is_menu_open() then
		cmd.in_attack = 0
        cmd.in_attack2 = 0
	end
    if lua_items.misc.aspectratio:get() then
        cvar.r_aspectratio:set_float(lua_items.misc.aspectratio_value:get()/100 or 0)
    end
    if lua_items.misc.thirdperson:get() then
        cvar.cam_idealdist:set_float(lua_items.misc.thirdperson_value:get() or 150)
    end

    gpt_peek( cmd )
end)

lua_items.misc.aspectratio:set_callback(function(self)
    if not self:get() then
        cvar.r_aspectratio:set_float(0)
    end
end)

lua_items.misc.thirdperson:set_callback(function(self)
    if not self:get() then
        cvar.cam_idealdist:set_float(150)
    end
end)

client.set_event_callback('paint', function()
    local lp = entity.get_local_player()
    if not lp then return end
    if not entity.is_alive(lp) then return end

    if lua_items.ragebot.scout_airstop:get() and lua_items.ragebot.scout_airstop_bind:get() then
        renderer.indicator(255, 255, 255, 255, 'AIrstop')
    end

    if lua_items.visuals.damage_ind:get() then
        visuals_helpers.damage_override()
    end
    if lua_items.visuals.velocity_warning:get() then
        visuals_helpers.velocity_ind()
    end
    if lua_items.visuals.defensive_warning:get() then
        visuals_helpers.defensive_ind()
    end
    if lua_items.visuals.manual_ind:get() then
        visuals_helpers.manual_arrows()
    end
    if lua_items.ragebot.predict:get() and lua_items.ragebot.predict_key:get() then
        renderer.indicator(200, 200, 200, 255, 'EX')
    end

    if lua_items.visuals.custom_scope:get() then
        visuals_helpers.custom_scope()
        ui.set(ref.scope, false)
    end

    if lua_items.visuals.debug_panel:get() then
        visuals_helpers.debug_panel()
    end

    if lua_items.visuals.kibit_hitmarker:get() then
        visuals_helpers.kibit_mark()
    end

    if lua_items.visuals.bullet_tracers:get() then
        visuals_helpers.bullet_tracers()
    end

    if lua_items.misc.auto_tp:get() and lua_items.misc.auto_tp_key:get() then
        if tp_state == 'ACT' then
            renderer.indicator(200, 200, 200, 255, 'LC')
        elseif tp_state == 'TP' then
            renderer.indicator(0, 200, 0, 255, 'LC')
        else
            renderer.indicator(200, 0, 0, 255, 'LC')
        end
    end

    if lua_items.ragebot.ai_peek:get() then
        render_peeking_points()
        if not ui.get(quick_peek_assist[ 2 ]) then return end
        get_closest_enemy()
        render_screen_bar()
    end
    local user = {} do
        local local_player = entity.get_local_player() 
        user.name = entity.get_player_name(local_player) 
        user.version = "Asaka ~ " .. user.name
    end
    
    renderer.text(center[1], screen[2] - 20, 255, 255, 255, 225,  "cd", 0, user.version)
end)

lua_items.visuals.custom_scope:set_callback(function(self)
    if self:get() then
        ui.set(ref.scope, true)
        s_length = 0
    end
end)

client.set_event_callback('paint_ui', function()
    lua_items.main.label1:set('\aB0B0B0FF「✦ '..text_fade_menu(-1, {r=200, g=200, b=200, a=200}, {r=150, g=150, b=150, a=150}, 'Asaka ~ pre release')..' ✦」')
    hide_original_menu(false)
    if lua_items.visuals.custom_scope:get() then
        ui.set(ref.scope, true)
    end
    misc_helpers.ragebot_logs()
    if lua_items.misc.clantag:get() then
        misc_helpers.set_clan()
    end
end)

local pred_state = false

client.set_event_callback('pre_render', function()
    if lua_items.ragebot.predict:get() and lua_items.ragebot.predict_key:get() then
        rage_helpers.predict()
        pred_state = true
    elseif pred_state == true then
        cvar.cl_interp:set_int(0.015625)
        cvar.cl_interpolate:set_int(1)
        cvar.cl_interp_ratio:set_int(2)
        cvar.cl_clock_correction:set_int(1)
        pred_state = false
    end
end)


--[[
local lol_pred_state = false

client.set_event_callback('paint', function()
    if username == 'frauch' then return end
    if lua_items.ragebot.predict:get() and lua_items.ragebot.predict_key:get() then
        if is_vulnerable() and not lol_pred_state then
            client.exec('sm_Lopgybetterthanall 1')
            client.delay_call(0.1, function()
                client.exec('clear')
            end)
            lol_pred_state = true
        elseif not is_vulnerable() and lol_pred_state then
            client.delay_call(0.2, function()
                client.exec('sm_Lopgybetterthanall 0')
            end)
            client.delay_call(0.3, function()
                client.exec('clear')
            end)
            lol_pred_state = false
        end
    elseif lol_pred_state then
        client.exec('sm_Lopgybetterthanall 0')
        client.delay_call(0.1, function()
            client.exec('clear')
        end)
        lol_pred_state = false
    end
end)

client.set_event_callback('round_start', function()
    if username == 'frauch' then return end
    client.exec('sm_Lopgybetterthanall 0')
    client.delay_call(0.1, function()
        client.exec('clear')
    end)
    lol_pred_state = false
end)

client.set_event_callback('shutdown', function()
    if username == 'frauch' then return end
    client.exec('sm_Lopgybetterthanall 0')
end)
]]


client.set_event_callback('shutdown', function()
    local played_seconds = client.unix_time() - start_time
    user_info[3] = user_info[3] + played_seconds
    database.write(lua_db.user, user_info)
    database.flush()
    cvar.cl_interp:set_int(0.015625)
    cvar.cl_interpolate:set_int(1)
    cvar.cl_interp_ratio:set_int(2)
    cvar.cl_clock_correction:set_int(1)
    cvar.cam_idealdist:set_int(150)
    cvar.r_aspectratio:set_float(0)
    cvar.con_filter_enable:set_int(0)
    cvar.con_filter_text:set_string("")
    cvar.viewmodel_offset_x:set_float(view_cvars.viewmodel_offset_x)
    cvar.viewmodel_offset_y:set_float(view_cvars.viewmodel_offset_y)
    cvar.viewmodel_offset_z:set_float(view_cvars.viewmodel_offset_z)
    cvar.fov_cs_debug:set_int(0)

    cvar.r_3dsky:set_int(fps_cvars.r_3dsky)

    cvar.fog_enable:set_int(fps_cvars.fog_enable)
    cvar.fog_enable_water_fog:set_int(fps_cvars.fog_enable_water_fog)
    cvar.fog_enableskybox:set_int(fps_cvars.fog_enableskybox)

    cvar.r_shadows:set_int(fps_cvars.r_shadows)

    cvar.violence_hblood:set_int(fps_cvars.violence_hblood)
    cvar.violence_ablood:set_int(fps_cvars.violence_ablood)

    cvar.r_decals:set_int(fps_cvars.r_decals)

    cvar.mat_postprocess_enable:set_int(fps_cvars.mat_postprocess_enable)

    cvar.cl_disable_ragdolls:set_int(fps_cvars.cl_disable_ragdolls)

    cvar.r_eyegloss:set_int(fps_cvars.r_eyegloss)
    cvar.r_eyemove:set_int(fps_cvars.r_eyemove)
    cvar.r_eyeshift_x:set_int(fps_cvars.r_eyeshift_x)
    cvar.r_eyeshift_y:set_int(fps_cvars.r_eyeshift_y)
    cvar.r_eyeshift_z:set_int(fps_cvars.r_eyeshift_z)
    cvar.r_eyesize:set_int(fps_cvars.r_eyesize)

    cvar.r_drawparticles:set_int(fps_cvars.r_drawparticles)

    cvar.cl_detail_avoid_radius:set_int(fps_cvars.cl_detail_avoid_radius)
    cvar.cl_detail_max_sway:set_int(fps_cvars.cl_detail_max_sway)
    cvar.dsp_slow_cpu:set_int(fps_cvars.dsp_slow_cpu)
    cvar.func_break_max_pieces:set_int(fps_cvars.func_break_max_pieces)
    cvar.r_drawtracers:set_int(fps_cvars.r_drawtracers)
    cvar.r_dynamic:set_int(fps_cvars.r_dynamic)
    cvar.muzzleflash_light:set_int(fps_cvars.muzzleflash_light)
    cvar.mat_hdr_enabled:set_int(fps_cvars.mat_hdr_enabled)

    client.exec("con_filter_enable 0")
    hide_original_menu(true)
    misc_helpers.fps_boost(false)
    client.set_clan_tag('')
    database.write(lua_db.config, configs_db)
end)

client.set_event_callback('on_player_spawn', function(e)
    reset_reso_date(e)
end)

client.set_event_callback('net_update_end', function()
    if lua_items.ragebot.resolver:get() then
        rage_helpers.jitter_resolver()
    end
end)

client.set_event_callback("level_init", function()
    alive_players = {}
    aa_helpers.reset_def()
    queue = {}
    queue2 = {}
    misc_helpers.filter(lua_items.misc.filter_console:get())
end)

client.set_event_callback("round_start", function()
    alive_players = {}
    aa_helpers.reset_def()
    rage_helpers.erase_player_data()
    rage_helpers.reset_plist()
    queue = {}
    queue2 = {}
    misc_helpers.filter(lua_items.misc.filter_console:get())
end)

client.set_event_callback("player_say", function(e)
    if lua_items.misc.enemy_chat:get() then
        misc_helpers.onPlaySay(e)
    end
end)

client.set_event_callback("player_chat", function(e)
    if lua_items.misc.enemy_chat:get() then
        misc_helpers.onPlayChat(e)
    end
end)

client.set_event_callback("round_end_upload_stats", misc_helpers.buybot)

local zoom_offset = 0

client.set_event_callback('override_view', function(e)
    local lp = entity.get_local_player()
    if not lp then return end
    if not lua_items.misc.animated_zoom:get() then return end

    local scoped = entity.get_prop(lp, "m_bIsScoped") == 1
    local weapon = entity.get_player_weapon(lp)
    local zoomlevel = entity.get_prop(weapon, "m_zoomLevel")

    local value = 0

    if scoped then
        if zoomlevel == 1 then
            value = lua_items.misc.animated_step:get()
        else
            value = lua_items.misc.animated_step:get() * 2
        end
    else
        value = 0
    end

    zoom_offset = math.lerp(zoom_offset, scoped and value or 0, 0.2)

    e.fov = ui.get(ref.fov) - zoom_offset
end)

local phrases_kill = {
    [1] = {'1','?'},
    [2] = {'bot?'},
    [3] = {'我没想到你的参数这么垃圾....'},
    [4] = {'今必斩汝马下！下一个是谁？'},
    [5] = {'尔等休走！枪管已为你奏响终曲'},
    [6] = {'1','死亡是我送你的唯一礼物，签收吧！'},
    [7] = {'棺材板压不住了？我来帮你钉牢！'},
    [8] = {'下一个！'},
    [9] = {'就这？'},
    [10] = {'键鼠撒米，鸡都比你秀'},
    [11] = {'?'},
    [12] = {'广场舞大妈端狙都比你e'},
    [13] = {'lc'},
    [14] = {'1','как на этот раз оправдаешься?'},
    [15] = {'0 iq'},
    [16] = {'спи','мача ебаная'},
    [17] = {'зря ты так летишь','у тебя ноль шансов убить меня'},
    [18] = {'понадеялся на удачу?'},
    [19] = {'парашыч ебанный'},
    [20] = {'куда ты пикаешь то','скряга ебаная'},
    [21] = {'1','пора ливать','чмошница'},
    [22] = {'пора ливать чмоня'},
    [23] = {'1','наивный ботик','куда пикаем'},
    [24] = {'пикнул?','сиди и наблюдай теперь чмо'},
    [25] = {'сука не позорься и ливни лол'},
    [26] = {'1','что ты делаешь?'},
    [27] = {'1','ахаха','спать шлюшка'},
    [28] = {'депортирован в ад к матери шлюхе'},
    [29] = {'1','скули сука'},
    [30] = {'поймал в шляпу?'},
    [31] = {'1','?','чурбек ебаный куда летим'},
    [32] = {'лови в пиздак мразота'},
    [33] = {'ez?'},
    [34] = {'игрок?'},
    [35] = {'изи упал нищий'},
    [36] = {'и это игрок?'},
    [37] = {'трахнут'},
    [38] = {'HAHAHAHAHHAHA','1 ДЕРЕВО ЕБАННОЕ'},
    [39] = {'куда ты пикаешь?','ты просто нулячий дядь'},
    [40] = {'опять забайтился мусор'},
    [41] = {'куда бежиш червяк'},
    [42] = {'отлетаешь','сын бляди'},
    [43] = {'1','оправдайся','почему ты опять умер'},
    [44] = {'хааххаха','опять умер'},
    [45] = {'1 мразота'}
}

local phrases_dead = {
    [1] = {'xx克高手'},
    [2] = {'Lukcy dog'},
    [3] = {'我在游戏你被你杀死，你妈在现实中被我操死'},
    [4] = {'听说当年你妈也是这样死的'},
    [5] = {'блядина','нахуй ты сидишь потееш','выйди в кд грязб'},
    [6] = {'щас поиграем клоун','ливнешь мать шалава здохнет'},
    [7] = {'фу','ну он же байтится','и както убивает меня'},
    [8] = {'отмена','сын шлюхи'},
    [9] = {'хаахахах','что ты нахуй делаешь'},
    [10] = {'фу'},
    [11] = {'нет','хуесос тупой'},
    [12] = {'опять','меня убивает','тупорылый','фу блять'},
    [13] = {'без пота слабо хуйня?'},
    [14] = {'мразота потеет'},
    [15] = {'фу уебище ебаное'},
    [16] = {'я струльнул?'},
    [17] = {'изи мапа'},
    [18] = {'м','забавно'},
    [19] = {'ну','что ты дулаешь'},
    [20] = {'ебанат','фулл опен стоит'},
    [21] = {'тупой','куда ты летишь идиот'},
    [22] = {'ну конечно','я просто похлопаю тебе'},
    [23] = {'не ливай'},
    [24] = {'сыну бляди же повезёт'},
    [25] = {'опять чмо ебаное убивает'},
    [26] = {'ну фу','тебе повезло выблядок'},
    [27] = {'ну фу','что ты делаеш'},
    [28] = {'подловила мразь'},
    [29] = {'не','как эьа хуйня убивает','это пиздец'},
    [30] = {'не','этот сочник пикнет'},
    [31] = {'csy','сын шлюхи','тупой'},
    [32] = {'что ты сделал','тупой даун','безмозглый'},
    [33] = {',kznm','тупой долбаеб','реально','уебище'},
    [34] = {'csy ik.[b','сын бляди ебаной','потеет сидит'},
    [35] = {'хуя','норм бектрек'},
    [36] = {'пиздец','что с читом'},
    [37] = {'красава','лучший просто'},
    [38] = {'ну','долбаеб сука','что ты делаешь мразь'}
}

client.set_event_callback("player_death", function(e)
    if not lua_items.misc.trashtalk:get() then return end
    local victim_userid, attacker_userid = e.userid, e.attacker
    if victim_userid == nil or attacker_userid == nil then
        return
    end

    local victim_entindex = client.userid_to_entindex(victim_userid)
    local attacker_entindex = client.userid_to_entindex(attacker_userid)

    local function send_delayed_messages(phrases)
        for i, phrase in ipairs(phrases) do
            client.delay_call(i*2, function() client.exec("say " .. phrase) end)
        end
    end

    if attacker_entindex == entity.get_local_player() and entity.is_enemy(victim_entindex) then
        local phrase_set = phrases_kill[math.random(1, #phrases_kill)]
        send_delayed_messages(phrase_set)
    end

    if attacker_entindex ~= entity.get_local_player() and victim_entindex == entity.get_local_player() then
        local phrase_set = phrases_dead[math.random(1, #phrases_dead)]
        send_delayed_messages(phrase_set)
    end
end)


client.set_event_callback('aim_miss', aim_miss)
client.set_event_callback('aim_hit', aim_hit)
client.set_event_callback('aim_fire', function(e)
    cache.last_seen = 0
    console_logs.on_aim_fire(e)
end)
client.set_event_callback('aim_miss', console_logs.on_aim_miss)
client.set_event_callback('aim_hit', console_logs.on_aim_hit)

client.set_event_callback("pre_render", function()
    if not lua_items.misc.anims:get() then return end
    local self = entity.get_local_player()
    if not self then return end
    local self_index = c_entity.new(self)
    local self_anim_state = self_index:get_anim_state()
    if not self_anim_state then return end
    local x_velocity = entity.get_prop(self, "m_vecVelocity[0]")

    if lua_items.misc.anims_ground:get() == 'Static' then
        entity.set_prop(entity.get_local_player(), "m_flPoseParameter", lua_items.misc.anims_ground_amount:get()/100, 0)
        ui.set(ref.legs, 'Always Slide')
    elseif lua_items.misc.anims_ground:get() == 'Jitter' then
        entity.set_prop(entity.get_local_player(), "m_flPoseParameter", globals.tickcount() % 4 > 1 and lua_items.misc.anims_ground_amount:get()/100 or 1, 0)
        ui.set(ref.legs, 'Always Slide')
    elseif lua_items.misc.anims_ground:get() == 'Jitter Slide' then
        entity.set_prop(entity.get_local_player(), "m_flPoseParameter", math.random(lua_items.misc.anims_ground_amount:get(), 100)/100, 0)
        ui.set(ref.legs, 'Always Slide')
    elseif lua_items.misc.anims_ground:get() == 'MoonWalk' then
        entity.set_prop(entity.get_local_player(), "m_flPoseParameter", 1, 7)
        ui.set(ref.legs, 'Never Slide')
    end

    if lua_items.misc.anims_air:get() == 'Static' then
        entity.set_prop(entity.get_local_player(), "m_flPoseParameter", lua_items.misc.anims_air_amount:get()/100, 6)
    elseif lua_items.misc.anims_air:get() == 'Jitter' then
        entity.set_prop(entity.get_local_player(), "m_flPoseParameter", globals.tickcount() % 4 > 1 and lua_items.misc.anims_air_amount:get()/100 or 1, 6)
    elseif lua_items.misc.anims_air:get() == 'Better Jitter' then
        entity.set_prop(entity.get_local_player(), "m_flPoseParameter", math.random(lua_items.misc.anims_air_amount:get(), 100)/100, 6)
        entity.set_prop(entity.get_local_player(), "m_flPoseParameter", math.random(0, 10)/10, 3)
    elseif lua_items.misc.anims_air:get() == 'MoonWalk' then
        local self_anim_overlay_6 = self_index:get_anim_overlay(6)
        if not self_anim_overlay_6 then return end
    
        if math.abs(x_velocity) >= 3 then
            self_anim_overlay_6.weight = 1
        end
    end

    local self_anim_overlay = self_index:get_anim_overlay(12)
    if not self_anim_overlay then return end

    if lua_items.misc.move_lean:get() == 'Static' then
        local x_velocity = entity.get_prop(self, "m_vecVelocity[0]")
        if math.abs(x_velocity) >= 3 then
            self_anim_overlay.weight = lua_items.misc.move_lean_amount:get()/100
        end
    elseif lua_items.misc.move_lean:get() == 'Jitter' then
        self_anim_overlay.weight = math.random(lua_items.misc.move_lean_amount:get()/100, 100)/100
    end
end)

local ws_data = {ws = nil}

local online_configs = {}

local function update_likes_label()
    local selected_index = lua_items.config.online_cfg:get() + 1--Хз не ворк кфг система щас
    if online_configs[selected_index] then
        local cfg = online_configs[selected_index]
        lua_items.config.label_like:set("  \v\r  Likes: " .. cfg.likes .. " | \v\r  Dislikes: " .. cfg.dislikes)
    end
end

lua_items.config.online_cfg:set_callback(function()
    update_likes_label()
end)

local function safe_send(data)
    if ws_data.ws == nil then
        print('You are not connected to the server')
        return
    end
    ws_data.ws:send(json.stringify(data))
end

lua_items.config.load_cfg:set_callback(function()
    local selected_index = lua_items.config.online_cfg:get() + 1
    if online_configs[selected_index] then
        safe_send({action = 'get_cfg', name = online_configs[selected_index].name})
    end
end)

--[[lua_items.config.upload_cfg:set_callback(function()
    local config_to_send = base64.encode(json.stringify(package:save()))
    local parts = {}
    local max_size = 7500
    for i = 1, #config_to_send, max_size do
        table.insert(parts, config_to_send:sub(i, i + max_size - 1))
    end

    for i, part in ipairs(parts) do
        safe_send({action = "upload_cfg_part", name = username, part_id = i, content = part})
    end

    safe_send({action = "upload_cfg_last", name = username})
end)]]


lua_items.config.like_button:set_callback(function()
    local selected_index = lua_items.config.online_cfg:get() + 1
    if online_configs[selected_index] then
        safe_send({action = 'like', name = online_configs[selected_index].name, username = username})
    end
end)

lua_items.config.dislike_button:set_callback(function()
    local selected_index = lua_items.config.online_cfg:get() + 1
    if online_configs[selected_index] then
        safe_send({action = 'dislike', name = online_configs[selected_index].name, username = username})
    end
end)

local received_parts = ''


local ws_callbacks = {
    open = function(ws)
        print('Connected to Server')
        ws_data.ws = ws
        safe_send({action = 'get_configs'})
    end,

    message = function(ws, data)
        local success, response = pcall(json.parse, data)
        if not success then
            print("Error parsing JSON:", response)
            return
        end

        if response.file_content_last then  
            local full_config = received_parts
            received_parts = ''
    
            local success, err = pcall(function()
                package:load(json.parse(base64.decode(full_config)))
            end)
            if not success then
                print("Error loading config:", err)
            end
        else
            for key, value in pairs(response) do
                if key:match("^file_content_%d+$") then
                    local part_id = tonumber(key:match("%d+"))
                    received_parts = received_parts .. value
                end
            end
        end
    

        if response.configs then
            online_configs = {}
            local config_names = {}

            local name_map = {
                Lopgy = '♛ Lopgy ♛'
            }
        
            for _, cfg in ipairs(response.configs) do
                table.insert(online_configs, {
                    name = cfg.name,
                    likes = cfg.likes or 0,
                    dislikes = cfg.dislikes or 0
                })

                if name_map[cfg.name] then
                    table.insert(config_names, name_map[cfg.name])
                else
                    table.insert(config_names, cfg.name)
                end
            end
        
            if #config_names > 0 then
                lua_items.config.online_cfg:update(config_names)
                update_likes_label()
            end   
        elseif response.status and response.name then
            print(response.status)
            for _, cfg in ipairs(online_configs) do
                if cfg.name == response.name then
                    cfg.likes = response.likes
                    cfg.dislikes = response.dislikes
                    update_likes_label()
                    break
                end
            end
        elseif response.status then
            print(response.status)
        end

        if response.user_count then
            lua_items.main.labelonline:set(' \v\r  Online Users: \v 2')
        end
    end,

    close = function(ws)
        print('Welcome! ' .. user.name )
    end,

    error = function(ws)
        print('Current version: 1.0')
    end
}


local success, result = pcall(function()
    return websockets.connect('ws://94.159.102.30:8833', ws_callbacks)
end)

if not success then
    print("Can't connect to server. Cloud configs and online users will not work", result)

end




