-- Part 1: Initial Setup, FFI, and UI Elements
local ffi = require("ffi")
local ent_c = { get_client_entity = vtable_bind('client.dll', 'VClientEntityList003', 3, 'void*(__thiscall*)(void*, int)') }
writefile("resolver_data.txt", "")



ffi.cdef[[
    typedef struct {
        char pad_0[0x18];
        float m_flFeetSpeedForwardsOrSideways;
        char pad_1C[0xC];
        float m_flStopToFullRunningFraction;
        float m_flDuckAmount;
        char pad_2C[0x74];
        float m_flMoveWeight;
        float m_flStrafeWeight;
        float m_flUnknownVelocityLean;
        char pad_AC[0x4];
        float m_flLadderSpeed;
        char pad_B4[0x4C];
        float m_flSpeed2D;
        float m_flUpVelocity;
        float m_flSpeedNormalized;
        float m_flFeetSpeedForwardsOrSideways;
        float m_flFeetSpeedUnknownForwardsOrSideways;
        float m_flTimeSinceStartedMoving;
        float m_flTimeSinceStoppedMoving;
        bool m_bOnGround;
        bool m_bHitGroundAnimation;
        char pad_135[0x4];
        float m_flLastOriginZ;
        float m_flHeadHeight;
        float m_flStopToFullRunningFraction;
        char pad_148[0x8];
        float m_flLeanYaw;
        char pad_154[0x8];
        float m_flPosesSpeed;
        char pad_160[0x8];
        float m_flLadderSpeed;
        char pad_16C[0x8];
        float m_flLadderYaw;
        char pad_178[0x8];
        float m_flEyeYaw;
        float m_flBodyYaw;
        float m_flGoalFeetYaw;
        float m_flBodyPitch;
        char pad_188[0x48];
        float m_flVelocitySubtractX;
        float m_flVelocitySubtractY;
        float m_flVelocitySubtractZ;
    } animation_state_t;

    typedef struct {
        char pad_0[0x18];
        uint32_t m_nSequence;
        float m_flPrevCycle;
        float m_flWeight;
        float m_flWeightDeltaRate;
        float m_flPlaybackRate;
        float m_flCycle;
    } animation_layer_t;
]]

-- Cache frequently used functions for performance
local band = bit.band
local sqrt = math.sqrt
local abs = math.abs
local deg = math.deg
local rad = math.rad
local exp = math.exp
local random = math.random
local floor = math.floor


local lbl1 = ui.new_label("LUA", "A", "----------------------\aBBC0F3FF Asaka DLC \aE8E8E8FF----------------------")
local resolver_master = ui.new_checkbox("LUA", "A", "Evade Asaka Resolver", true)
local resolver_debug = ui.new_checkbox("LUA", "A", "Show Devlog")
local lbl2 = ui.new_label("LUA", "A", "----------------------\aBBC0F3FF-----------\aE8E8E8FF----------------------")


-- Constants and Layer Definitions (keeping original)
local RESOLVER_LAYERS = {
    MOVEMENT = 7,
    ADJUST = 12,
    LEAN = 3,
    JUMP = 4,
    ALIVELOOP = 8
}

local POSE_PARAMS = {
    BODY_YAW = 11,
    BODY_PITCH = 12,
    MOVE_YAW = 15,
    STAND = 38,
    CROUCH_AMOUNT = 36
}

local MOVEMENT_FLAGS = {
    ON_LADDER = 8,
    SWIMMING = 9,
    FL_ONGROUND = 9
}

-- Enhanced Resolver Data Structure
local ResolverData = {
    successful_hits = {},
    save_interval = 10,
    hit_counter = 0,
    quality_metrics = {
        accuracy_threshold = 0.7,
        confidence_threshold = 0.8,
        minimum_samples = 32
    },
    historical_performance = {},
    data_validation = {
        min_desync = -58,
        max_desync = 58,
        max_eye_yaw = 180,
        min_eye_yaw = -180
    }
}

-- Global variables (keeping original)
local meta_jitter_players = {}
local animlayer_average_t = {}
local last_shot_time = {}
local anti_aim_signatures = {
    meta_jitter = {},
    rapid_manipulation = {},
    player_specific_patterns = {}
}
local config = {
    isnt_defensive = false
}
-- Weak table cache creation (keeping original)
local function CreateWeakCache()
    return setmetatable({}, {__mode = "k"})
end

-- Global Caches (keeping original)
local Records = CreateWeakCache()
local resolver_states = CreateWeakCache()
local hit_miss_data = CreateWeakCache()
local last_pitch = CreateWeakCache()
local pitch_history = CreateWeakCache()
local previous_yaw = CreateWeakCache()

-- Debug Configuration (keeping original)
local PITCH_HISTORY_SIZE = 8
local PITCH_CHANGE_THRESHOLD = 45
local PITCH_FAKE_FLICK_THRESHOLD = 60
local debug_font_flags = "b-d"
local debug_x = 10
local debug_y_start = 150
local debug_line_height = 9
local debug_alpha = 255

-- Part 2: Utility Functions and Enhanced Neural Network Core

-- Utility Functions (keeping original)
local function Clamp(value, min, max)
    return math.min(math.max(value, min), max)
end

local function NormalizeAngle(angle)
    while angle > 180 do angle = angle - 360 end
    while angle < -180 do angle = angle + 360 end
    return angle
end

local function VectorAngle(x, y)
    return deg(math.atan2(y, x))
end

local function IsValidEntity(ent)
    if not ent or not entity.is_alive(ent) then return false end
    if entity.get_classname(ent) ~= "CCSPlayer" then return false end
    
    local localplayer = entity.get_local_player()
    return localplayer and entity.is_alive(localplayer) and entity.get_prop(ent, "m_iTeamNum") ~= entity.get_prop(localplayer, "m_iTeamNum")
end


local function SafeAnimstate(ent)
    local ent_ptr = ent_c.get_client_entity(ent)
    return ent_ptr ~= ffi.NULL and ffi.cast("animation_state_t*", ffi.cast("char*", ent_ptr) + 0x9960) or nil
end

local function SafeAnimLayer(ent, layer)
    local ent_ptr = ent_c.get_client_entity(ent) 
    return ent_ptr ~= ffi.NULL and ffi.cast("animation_layer_t*", ffi.cast('char*', ent_ptr) + 0x2990)[layer] or nil
end


-- Animation Layer and Pose Parameter Correction
local function FixAnimationLayers(ent)
    for layer, data in pairs(RESOLVER_LAYERS) do
        local l = SafeAnimLayer(ent, data)
        if l then
            if layer == "MOVEMENT" and l.m_flCycle > 0.9 then
                entity.set_prop(ent, "m_flCycle", 0.0, data)
            end
            
            if layer == "ADJUST" and l.m_flWeight < 0.1 then
                entity.set_prop(ent, "m_flWeight", 0.0, data)
            end
            
            if layer == "LEAN" and l.m_nSequence ~= 0 then
                entity.set_prop(ent, "m_nSequence", 0, data)
            end
        end
    end
end


local function CorrectPoseParameters(ent)
    local state = SafeAnimstate(ent)
    if not state then return end

    local pitch = entity.get_prop(ent, "m_angEyeAngles[0]") or 0
    entity.set_prop(ent, "m_flPoseParameter", Clamp(pitch/180, 0, 1), POSE_PARAMS.BODY_PITCH)

    local velocity = {entity.get_prop(ent, "m_vecVelocity")}
    local move_yaw = deg(math.atan2(velocity[2], velocity[1]))
    entity.set_prop(ent, "m_flPoseParameter", (move_yaw % 360) / 360, POSE_PARAMS.MOVE_YAW)
end

-- Movement and State Validation Functions
local function GetMoveType(ent)
    local flags = entity.get_prop(ent, "m_fFlags") or 0
    return {
        on_ladder = band(flags, MOVEMENT_FLAGS.ON_LADDER) ~= 0,
        swimming = band(flags, MOVEMENT_FLAGS.SWIMMING) ~= 0,
        on_ground = band(flags, MOVEMENT_FLAGS.FL_ONGROUND) ~= 0
    }
end



-- Enhanced Neural Network Configuration
local NN = {
    input_size = 16,
    hidden_size = 32,
    hidden_layers = 2,
    output_size = 3,
    learning_rate = 0.1,  -- Increased from 0.04
    momentum = 0.92,
    weights = {
        hidden = {},
        hidden2 = {}, -- New second hidden layer
        output = {},
        hidden_momentum = {},
        hidden2_momentum = {}, -- Momentum for second layer
        output_momentum = {}
    },
    batch_size = 64,
    training_buffer = {},
    performance_metrics = {}, -- Track performance
    data_quality = {}, -- Track data quality metrics
    prediction_history = {}
}


-- Enhanced Neural Network Core Functions
local function InitializeEnhancedWeights()
    math.randomseed(globals.curtime() * 1000)
    
    -- Initialize first hidden layer
    for i = 1, NN.input_size do
        NN.weights.hidden[i] = {}
        NN.weights.hidden_momentum[i] = {}
        for j = 1, NN.hidden_size do
            NN.weights.hidden[i][j] = (random() * 2 - 1) * sqrt(2 / (NN.input_size + NN.hidden_size))
            NN.weights.hidden_momentum[i][j] = 0
        end
    end
    
    -- Initialize second hidden layer
    for i = 1, NN.hidden_size do
        NN.weights.hidden2[i] = {}
        NN.weights.hidden2_momentum[i] = {}
        for j = 1, NN.hidden_size do
            NN.weights.hidden2[i][j] = (random() * 2 - 1) * sqrt(2 / (NN.hidden_size + NN.hidden_size))
            NN.weights.hidden2_momentum[i][j] = 0
        end
    end
    
    -- Initialize output layer
    for i = 1, NN.hidden_size do
        NN.weights.output[i] = {}
        NN.weights.output_momentum[i] = {}
        for j = 1, NN.output_size do
            NN.weights.output[i][j] = (random() * 2 - 1) * sqrt(2 / (NN.hidden_size + NN.output_size))
            NN.weights.output_momentum[i][j] = 0
        end
    end
end


-- Enhanced activation functions
local function sigmoid(x)
    return 1 / (1 + exp(-x))
end

local function relu(x)
    return x > 0 and x or 0
end

local function leaky_relu(x)
    return x > 0 and x or 0.01 * x
end






-- Enhanced forward pass with multiple layers
local function EnhancedForwardPass(inputs)
    -- First hidden layer
    local hidden1 = {}
    for j = 1, NN.hidden_size do
        local sum = 0
        for i = 1, NN.input_size do
            sum = sum + inputs[i] * NN.weights.hidden[i][j]
        end
        hidden1[j] = leaky_relu(sum)
    end
    
    -- Second hidden layer
    local hidden2 = {}
    for j = 1, NN.hidden_size do
        local sum = 0
        for i = 1, NN.hidden_size do
            sum = sum + hidden1[i] * NN.weights.hidden2[i][j]
        end
        hidden2[j] = leaky_relu(sum)
    end
    
    -- Output layer with more controlled output
    local outputs = {}
    for j = 1, NN.output_size do
        local sum = 0
        for i = 1, NN.hidden_size do
            sum = sum + hidden2[i] * NN.weights.output[i][j]
        end
        
        -- Use hyperbolic tangent for more balanced output
        outputs[j] = (math.tanh(sum) + 1) / 2  -- Maps to [0, 1]
    end
    
    return outputs, hidden1, hidden2
end

-- Enhanced backward pass with multiple layers
local function EnhancedBackwardPass(inputs, targets, predictions, hidden1, hidden2)
    local output_errors = {}
    local hidden2_errors = {}
    local hidden1_errors = {}
    
    -- Calculate output layer errors
    for i = 1, NN.output_size do
        output_errors[i] = predictions[i] * (1 - predictions[i]) * (targets[i] - predictions[i])
    end
    
    -- Calculate second hidden layer errors
    for i = 1, NN.hidden_size do
        hidden2_errors[i] = 0
        for j = 1, NN.output_size do
            hidden2_errors[i] = hidden2_errors[i] + output_errors[j] * NN.weights.output[i][j]
        end
        hidden2_errors[i] = hidden2_errors[i] * (hidden2[i] > 0 and 1 or 0.01) -- Leaky ReLU derivative
    end
    
    -- Calculate first hidden layer errors
    for i = 1, NN.hidden_size do
        hidden1_errors[i] = 0
        for j = 1, NN.hidden_size do
            hidden1_errors[i] = hidden1_errors[i] + hidden2_errors[j] * NN.weights.hidden2[i][j]
        end
        hidden1_errors[i] = hidden1_errors[i] * (hidden1[i] > 0 and 1 or 0.01) -- Leaky ReLU derivative
    end
    
    -- Update weights with momentum
    -- Output layer
    for i = 1, NN.hidden_size do
        for j = 1, NN.output_size do
            local delta = NN.learning_rate * output_errors[j] * hidden2[i]
            NN.weights.output[i][j] = NN.weights.output[i][j] + delta + NN.momentum * NN.weights.output_momentum[i][j]
            NN.weights.output_momentum[i][j] = delta
        end
    end
    
    -- Second hidden layer
    for i = 1, NN.hidden_size do
        for j = 1, NN.hidden_size do
            local delta = NN.learning_rate * hidden2_errors[j] * hidden1[i]
            NN.weights.hidden2[i][j] = NN.weights.hidden2[i][j] + delta + NN.momentum * NN.weights.hidden2_momentum[i][j]
            NN.weights.hidden2_momentum[i][j] = delta
        end
    end
    
    -- First hidden layer
    for i = 1, NN.input_size do
        for j = 1, NN.hidden_size do
            local delta = NN.learning_rate * hidden1_errors[j] * inputs[i]
            NN.weights.hidden[i][j] = NN.weights.hidden[i][j] + delta + NN.momentum * NN.weights.hidden_momentum[i][j]
            NN.weights.hidden_momentum[i][j] = delta
        end
    end
end

-- Initialize Neural Network weights
InitializeEnhancedWeights()


local function MonitorModelPerformance(ent)
    if not NN.performance_metrics[ent] then
        NN.performance_metrics[ent] = {
            hits = 0,
            misses = 0,
            total_error = 0,
            predictions = {},  -- Initialize empty table
            prediction_errors = {},  -- Add this for tracking
            last_update = globals.curtime(),
            last_confidence = 0
        }
    end
    
    local metrics = NN.performance_metrics[ent]
    
    -- Update metrics periodically
    if globals.curtime() - metrics.last_update > 1.0 then
        -- Ensure predictions exists and has values
        if not metrics.predictions then
            metrics.predictions = {}
        end
        
        -- Calculate average prediction error
        local avg_error = metrics.total_error / (math.max(#metrics.predictions, 1))
        
        -- Calculate hit rate
        local total_shots = metrics.hits + metrics.misses
        local hit_rate = total_shots > 0 and (metrics.hits / total_shots) or 0
        
        -- Reset counters
        metrics.total_error = 0
        metrics.predictions = {}
        metrics.last_update = globals.curtime()
        
        -- Adjust learning rate based on performance
        if hit_rate < 0.3 then
            NN.learning_rate = math.min(NN.learning_rate * 1.1, 0.05)
        elseif hit_rate > 0.7 then
            NN.learning_rate = math.max(NN.learning_rate * 0.9, 0.01)
        end
        
        return {
            hit_rate = hit_rate,
            avg_error = avg_error,
            learning_rate = NN.learning_rate
        }
    end
    
    return nil
end
-- Enhanced Data Quality Monitoring
local function MonitorDataQuality(ent)
    if not NN.data_quality[ent] then
        NN.data_quality[ent] = {
            samples = {},
            quality_scores = {},
            last_cleanup = globals.curtime()
        }
    end
    
    local quality_data = NN.data_quality[ent]
    
    -- Periodic data cleanup
    if globals.curtime() - quality_data.last_cleanup > 30.0 then
        -- Remove old or low-quality samples
        local new_samples = {}
        local new_scores = {}
        
        for i, sample in ipairs(quality_data.samples) do
            local score = quality_data.quality_scores[i]
            if score > 0.7 and globals.curtime() - sample.time < 60 then
                table.insert(new_samples, sample)
                table.insert(new_scores, score)
            end
        end
        
        quality_data.samples = new_samples
        quality_data.quality_scores = new_scores
        quality_data.last_cleanup = globals.curtime()
    end
    
    return quality_data
end
local function ProcessNeuralNetworkData(features)
    local processed_data = {}
    
    -- More precise normalization
    for key, value in pairs(features) do
        if type(value) == "number" then
            if key == "desync" then
                -- Normalize desync to [0, 1] range
                processed_data[#processed_data + 1] = (value + 58) / 116
            elseif key == "speed" then
                -- Normalize speed with lower cap
                processed_data[#processed_data + 1] = math.min(value / 300, 1)
            elseif key == "eye_pitch" or key == "body_yaw" or key == "move_yaw" then
                -- Angle normalization to [0, 1]
                processed_data[#processed_data + 1] = (value + 180) / 360
            else
                -- Fallback normalization
                processed_data[#processed_data + 1] = math.min(math.max(value, 0), 1)
            end
        end
    end 
    
    -- Ensure consistent input size
    while #processed_data < NN.input_size do    
        processed_data[#processed_data + 1] = 0 
    end

    return processed_data
end
-- Enhanced Neural Network Training Integration
local function UpdateNeuralNetworkModel(ent, was_hit, features, actual_angles)
    if not NN.training_results then NN.training_results = {} end
    if not NN.training_results[ent] then NN.training_results[ent] = {} end
    

    
    local processed_features = ProcessNeuralNetworkData(features)
    
    -- Normalize target angles
    local target_angles = {
        (actual_angles.eye_yaw + 180) / 360,
        (actual_angles.desync + 58) / 116,
        (actual_angles.pitch + 89) / 178
    }
    
    local predictions, hidden1, hidden2 = EnhancedForwardPass(processed_features)
    
    -- Calculate prediction error with more detailed logging
    local pred_error = 0
    local error_details = {}
    for i = 1, #predictions do
        local current_error = math.abs(predictions[i] - target_angles[i])
        pred_error = pred_error + current_error
        table.insert(error_details, current_error)
    end
    
    -- Detailed error logging

    
    -- Dynamic training criteria
    if pred_error > 0.3 or was_hit then  -- Lower threshold
        -- More aggressive training
        NN.learning_rate = math.min(NN.learning_rate * 1.2, 0.2)
        -- Force training on more samples
        EnhancedBackwardPass(processed_features, target_angles, predictions, hidden1, hidden2)
    
        table.insert(NN.training_results[ent], {
            pred_error = pred_error,
            was_hit = was_hit,
            time = globals.curtime(),
            features = features
        })
        
        -- Limit training buffer size
        if #NN.training_results[ent] > 64 then
            table.remove(NN.training_results[ent], 1)
        end
        
        -- Success rate calculation
        local recent_success = 0
        for _, result in ipairs(NN.training_results[ent]) do
            if result.pred_error < 0.4 or result.was_hit then
                recent_success = recent_success + 1
            end
        end
        
        NN.success_rate = recent_success / math.max(#NN.training_results[ent], 1)
        

    end
    
    return pred_error, NN.success_rate
end
local function AssessFeatureQuality(features)
    local quality = 1.0
    
    -- Check for invalid values
    for key, value in pairs(features) do
        if value ~= value or value == nil then  -- NaN or nil check
            quality = quality * 0.8
        end
    end
    
    -- Check value ranges
    if features.desync and (features.desync < 0 or features.desync > 58) then
        quality = quality * 0.9
    end
    
    if features.speed and features.speed < 0 then
        quality = quality * 0.9
    end
    
    if features.duck and (features.duck < 0 or features.duck > 1) then
        quality = quality * 0.9
    end
    
    -- Check for movement abnormalities
    if features.speed > 0 and features.on_ground and features.playback_rate < 0.1 then
        quality = quality * 0.8  -- Suspicious movement
    end
    
    -- Check for animation coherence
    if features.playback_rate > 0 and features.speed < 1 then
        quality = quality * 0.9  -- Animation/movement mismatch
    end
    
    return quality
end

-- Part 3: Animation State, Layer Functions, and Neural Network Data Processing

-- Safe Animation State and Layer Functions (keeping original)
local function ExtractFeatures(ent)
    local velocity = {entity.get_prop(ent, "m_vecVelocity")}
    local speed = sqrt((velocity[1] or 0)^2 + (velocity[2] or 0)^2)
    local flags = entity.get_prop(ent, "m_fFlags") or 0
    local on_ground = band(flags, 1) ~= 0
    
    -- Get animation layer properties with enhanced validation
    local anim_layer_6 = SafeAnimLayer(ent, 6)
    local m_flPlaybackRate = anim_layer_6 and anim_layer_6.m_flPlaybackRate or 0

    -- Enhanced average calculation with validation
    animlayer_average_t[ent] = animlayer_average_t[ent] or {}
    if m_flPlaybackRate ~= 0 then  -- Only add valid rates
        table.insert(animlayer_average_t[ent], 1, m_flPlaybackRate)
        if #animlayer_average_t[ent] > 18 then
            table.remove(animlayer_average_t[ent])
        end
    end
    
    -- Enhanced feature set with validation
    local features = {
        desync = abs(NormalizeAngle(
            entity.get_prop(ent, "m_angEyeAngles[1]") - 
            ((entity.get_prop(ent, "m_flPoseParameter", POSE_PARAMS.BODY_YAW) or 0) * 116 - 58)
        )),
        speed = speed,
        playback_rate = m_flPlaybackRate,
        duck = entity.get_prop(ent, "m_flDuckAmount") or 0,
        on_ground = on_ground,
        air_time = on_ground and 0 or (resolver_states[ent] and resolver_states[ent].air_time or 0) + globals.tickinterval()
    }

    -- Add quality metrics
    features.quality = AssessFeatureQuality(features)
    
    return features
end

-- Enhanced Feature Extraction for Neural Network
local function ExtractEnhancedFeatures(ent)
    local velocity = {entity.get_prop(ent, "m_vecVelocity")}
    local speed = sqrt((velocity[1] or 0)^2 + (velocity[2] or 0)^2)
    local flags = entity.get_prop(ent, "m_fFlags") or 0
    local on_ground = band(flags, 1) ~= 0
    
    -- Initialize resolver state if it doesn't exist
    if not resolver_states[ent] then
        resolver_states[ent] = {
            air_time = 0,
            last_on_ground = true
        }
    end
    
    -- Update air time
    if not on_ground then
        resolver_states[ent].air_time = (resolver_states[ent].air_time or 0) + globals.tickinterval()
    else
        resolver_states[ent].air_time = 0
    end

    local adjust_layer = SafeAnimLayer(ent, RESOLVER_LAYERS.ADJUST)
    local move_layer = SafeAnimLayer(ent, RESOLVER_LAYERS.MOVEMENT)
    local lean_layer = SafeAnimLayer(ent, RESOLVER_LAYERS.LEAN)
    local jump_layer = SafeAnimLayer(ent, RESOLVER_LAYERS.JUMP)
    local animstate = SafeAnimstate(ent)
    
    return {
        desync = abs(NormalizeAngle(
            entity.get_prop(ent, "m_angEyeAngles[1]") - 
            ((entity.get_prop(ent, "m_flPoseParameter", POSE_PARAMS.BODY_YAW) or 0) * 116 - 58)
        )) / 60,
        speed = speed / 300,
        adjust_weight = adjust_layer.m_flWeight,
        move_weight = move_layer.m_flWeight,
        lean_weight = lean_layer.m_flWeight,
        jump_weight = jump_layer.m_flWeight,
        duck = entity.get_prop(ent, "m_flDuckAmount") or 0,
        on_ground = on_ground and 1 or 0,
        velocity_z = velocity[3] / 300,
        time_since_last_shot = globals.curtime() - (last_shot_time[ent] or 0),
        eye_pitch = (entity.get_prop(ent, "m_angEyeAngles[0]") or 0) / 90,
        body_yaw = animstate.m_flEyeYaw / 180,
        move_yaw = animstate.m_flEyeYaw / 180,
        air_time = resolver_states[ent].air_time,
        playback_rate = move_layer.m_flPlaybackRate,
        cycle = move_layer.m_flCycle
    }
end


local function AssessDataQuality(features, prediction, actual)
    local quality_score = 0
    
    -- Check feature completeness and validity
    local feature_count = 0
    for _, value in pairs(features) do
        if value ~= nil and type(value) == "number" and value == value then -- Check for NaN
            feature_count = feature_count + 1
        end
    end
    
    -- Feature completeness score (0.3)
    quality_score = quality_score + (feature_count / NN.input_size) * 0.3
    
    -- Prediction accuracy score (0.4)
    if prediction and actual then
        local error = abs(prediction - actual)
        local normalized_error = Clamp(1 - (error / 180), 0, 1)
        quality_score = quality_score + normalized_error * 0.4
    end
    
    -- Temporal relevance score (0.3)
    local time_factor = Clamp(1 - ((globals.curtime() - (features.time or globals.curtime())) / 30), 0, 1)
    quality_score = quality_score + time_factor * 0.3
    
    return quality_score
end



-- Enhanced movement prediction
local function PredictAdjustedVelocity(ent)
    local velocity = {entity.get_prop(ent, "m_vecVelocity")}
    local flags = entity.get_prop(ent, "m_fFlags") or 0
    local on_ground = band(flags, 1) ~= 0
    local duck_amount = entity.get_prop(ent, "m_flDuckAmount") or 0

    -- Enhanced air prediction
    if not on_ground then
        velocity[3] = velocity[3] - (globals.tickinterval() * 800)
        
        -- Air strafe prediction
        local eye_angles = {entity.get_prop(ent, "m_angEyeAngles")}
        if eye_angles[2] then
            local yaw_rad = rad(eye_angles[2])
            local strafe_factor = 0.52 + (duck_amount * 0.1)
            
            velocity[1] = velocity[1] + (math.cos(yaw_rad) * strafe_factor)
            velocity[2] = velocity[2] + (math.sin(yaw_rad) * strafe_factor)
        end
    end

    -- Enhanced ground movement prediction
    if on_ground then
        local speed = sqrt(velocity[1]^2 + velocity[2]^2)
        local friction = 0.92 -- Base friction
        
        -- Adjust friction based on duck state
        if duck_amount > 0 then
            friction = friction + (duck_amount * 0.03)
        end
        
        -- Apply friction with speed threshold
        if speed > 5 then
            velocity[1] = velocity[1] * friction
            velocity[2] = velocity[2] * friction
        end
    end

    return velocity
end


local function AnalyzeMovementPatterns(ent)
    local state = resolver_states[ent] or {}
    local features = ExtractFeatures(ent)
    
    state.movement_history = state.movement_history or {}
    table.insert(state.movement_history, {
        speed = features.speed,
        time = globals.curtime(),
        duck = features.duck,
        on_ground = features.on_ground,
        playback_rate = features.playback_rate
    })
    
    -- Keep history size manageable with data quality consideration
    while #state.movement_history > 64 do
        table.remove(state.movement_history, 1)
    end
    
    -- Enhanced pattern analysis
    local pattern = {
        direction_changes = 0,
        speed_variance = 0,
        is_strafing = false,
        duck_patterns = {},
        movement_consistency = 0
    }
    
    if #state.movement_history > 2 then
        local prev_speed = state.movement_history[#state.movement_history-1].speed
        local total_speed_delta = 0
        local consistent_movements = 0
        
        for i = #state.movement_history-1, 2, -1 do
            local curr = state.movement_history[i]
            local prev = state.movement_history[i-1]
            
            -- Enhanced speed analysis
            local speed_delta = abs(curr.speed - prev.speed)
            total_speed_delta = total_speed_delta + speed_delta
            
            if speed_delta > 15 then
                pattern.direction_changes = pattern.direction_changes + 1
            end
            
            -- Movement consistency check
            if abs(speed_delta - prev_speed) < 5 then
                consistent_movements = consistent_movements + 1
            end 
            
            -- Duck pattern analysis
            if curr.duck > 0.1 then
                table.insert(pattern.duck_patterns, {
                    amount = curr.duck,
                    time = curr.time
                })
            end
            
            prev_speed = curr.speed
        end
        
        pattern.speed_variance = total_speed_delta / (#state.movement_history - 1)
        pattern.movement_consistency = consistent_movements / (#state.movement_history - 2)
        pattern.is_strafing = pattern.direction_changes > 3 and pattern.speed_variance > 10
    end
    
    -- Update state with enhanced analysis
    resolver_states[ent] = state
    return pattern
end

-- Initialize the neural network data processing
local neural_network_data = {
    training_samples = {},
    validation_samples = {},
    recent_accuracy = {}
}

local function AnalyzeAnimationLayers(ent)
    local state = resolver_states[ent] or {}
    state.layer_history = state.layer_history or {
        adjust = {},  -- Layer 12
        movement = {},  -- Layer 6
        lean = {}     -- Layer 3
    }
    
    -- Get current layers
    local adjust_layer = SafeAnimLayer(ent, RESOLVER_LAYERS.ADJUST)
    local movement_layer = SafeAnimLayer(ent, RESOLVER_LAYERS.MOVEMENT)
    local lean_layer = SafeAnimLayer(ent, RESOLVER_LAYERS.LEAN)
    
    -- Store layer data with enhanced metrics
    table.insert(state.layer_history.adjust, {
        weight = adjust_layer.m_flWeight,
        playback = adjust_layer.m_flPlaybackRate,
        cycle = adjust_layer.m_flCycle,
        time = globals.curtime(),
        sequence = adjust_layer.m_nSequence
    })
    
    table.insert(state.layer_history.movement, {
        weight = movement_layer.m_flWeight,
        playback = movement_layer.m_flPlaybackRate,
        cycle = movement_layer.m_flCycle,
        time = globals.curtime(),
        sequence = movement_layer.m_nSequence
    })
    
    -- Keep history size manageable
    while #state.layer_history.adjust > 64 do
        table.remove(state.layer_history.adjust, 1)
        table.remove(state.layer_history.movement, 1)
    end
    
    -- Enhanced layer correlation analysis
    local desync_state = {
        is_max_desync = false,
        desync_side = 0,
        confidence = 0,
        pattern_strength = 0
    }
    
    -- Analyze patterns in adjust layer
    if #state.layer_history.adjust > 16 then
        local weight_changes = 0
        local high_weight_count = 0
        local pattern_score = 0
        
        for i = 2, #state.layer_history.adjust do
            local curr = state.layer_history.adjust[i]
            local prev = state.layer_history.adjust[i-1]
            
            -- Enhanced pattern detection
            if math.abs(curr.weight - prev.weight) > 0.5 then
                weight_changes = weight_changes + 1
                pattern_score = pattern_score + (1 - math.abs(curr.playback - prev.playback))
            end
            
            if curr.weight > 0.9 then
                high_weight_count = high_weight_count + 1
            end
        end
        
        desync_state.is_max_desync = high_weight_count > (#state.layer_history.adjust * 0.7)
        desync_state.pattern_strength = pattern_score / (#state.layer_history.adjust - 1)
        desync_state.confidence = weight_changes / #state.layer_history.adjust
    end
    
    -- Enhanced movement-adjust correlation analysis
    if #state.layer_history.movement > 16 then
        local move_adjust_correlation = 0
        local pattern_consistency = 0
        
        for i = 2, #state.layer_history.movement do
            local move = state.layer_history.movement[i]
            local adjust = state.layer_history.adjust[i]
            
            if move.playback > 0.8 and adjust.weight < 0.2 then
                move_adjust_correlation = move_adjust_correlation + 1
                
                -- Check pattern consistency
                if i > 2 then
                    local prev_move = state.layer_history.movement[i-1]
                    local prev_adjust = state.layer_history.adjust[i-1]
                    if math.abs((move.playback - prev_move.playback) - 
                               (adjust.weight - prev_adjust.weight)) < 0.1 then
                        pattern_consistency = pattern_consistency + 1
                    end
                end
            end
        end
        
        -- Update confidence with enhanced metrics
        desync_state.confidence = desync_state.confidence * 
            (1 + (move_adjust_correlation / #state.layer_history.movement)) *
            (1 + (pattern_consistency / (#state.layer_history.movement - 2)))
    end
    
    -- Store state with enhanced neural network integration
    state.nn_features = {
        adjust_pattern = desync_state.pattern_strength,
        move_correlation = desync_state.confidence,
        is_max_desync = desync_state.is_max_desync and 1 or 0
    }
    
    resolver_states[ent] = state
    return desync_state
end


local function DetectMetaJitter(ent)
    local features = ExtractFeatures(ent)
    

    local current_yaw = entity.get_prop(ent, "m_angEyeAngles[1]") or 0

    -- Store previous angles for comparison
    if not previous_yaw[ent] then
        previous_yaw[ent] = {
            angles = {},
            last_check = globals.curtime(),
            jitter_count = 0,
            last_yaw = current_yaw -- Store last valid yaw
        }
    end
    

    if current_yaw ~= 0 and current_yaw ~= previous_yaw[ent].last_yaw then
        table.insert(previous_yaw[ent].angles, current_yaw)
        previous_yaw[ent].last_yaw = current_yaw
    end
    

    if #previous_yaw[ent].angles > 6 then
        table.remove(previous_yaw[ent].angles, 1)
    end
    

    if #previous_yaw[ent].angles >= 2 then
        for i = 2, #previous_yaw[ent].angles do
            local delta = math.abs(NormalizeAngle(previous_yaw[ent].angles[i] - previous_yaw[ent].angles[i-1]))
            if delta > 120 then
                return "defensive"
            end
        end
    end

    local is_jittering = false
    if #previous_yaw[ent].angles >= 4 then
        local large_changes = 0
        local last_was_large = false
        
        for i = 2, #previous_yaw[ent].angles do
            local delta = math.abs(NormalizeAngle(previous_yaw[ent].angles[i] - previous_yaw[ent].angles[i-1]))
            
            
            -- Looking for alternating pattern of large changes and small/no changes
            if delta > 30 then
                if not last_was_large then
                    large_changes = large_changes + 1
                end
                last_was_large = true
            else
                last_was_large = false
            end
        end
        
        -- Consider it jittering if we see at least 2 large changes with alternating pattern
        is_jittering = large_changes >= 2
        
        if is_jittering then 
            previous_yaw[ent].jitter_count = previous_yaw[ent].jitter_count + 1
        else
            previous_yaw[ent].jitter_count = math.max(0, previous_yaw[ent].jitter_count - 1)
        end
    end
    
    -- Set jitter state if we've seen consistent jitter pattern
    if is_jittering and previous_yaw[ent].jitter_count >= 2 then
        meta_jitter_players[ent] = {
            detected_time = globals.curtime(),
            side = features.desync > 0 and 1 or -1,
            angle_changes = previous_yaw[ent].angles
        }
        
        -- Force a small desync value when jitter is detected
        features.desync = 10 * (features.desync > 0 and 1 or -1)
        
        return true
    end
    
    -- Clear state if no jitter for a while
    if previous_yaw[ent].jitter_count <= 0 then
        meta_jitter_players[ent] = nil
    end
    
    return false
end

local function CreateAntiAimSignature(ent)
    return {
        angle_history = {},
        desync_history = {},
        manipulation_count = 0,
        last_significant_change = 0,
        signature_confidence = 0,
        pattern_analysis = {
            frequent_angles = {},
            switch_times = {},
            consistency_score = 0
        }
    }
end


local function DetectRapidAngleManipulation(ent)
    local current_yaw = entity.get_prop(ent, "m_angEyeAngles[1]") or 0
    local current_time = globals.curtime()
    
    if not anti_aim_signatures[ent] then
        anti_aim_signatures[ent] = CreateAntiAimSignature(ent)
    end
    
    local signature = anti_aim_signatures[ent]
    
    -- Enhanced angle history tracking
    table.insert(signature.angle_history, {
        yaw = current_yaw,
        time = current_time,
        delta_time = signature.angle_history[#signature.angle_history] 
            and (current_time - signature.angle_history[#signature.angle_history].time) 
            or 0
    })
    
    -- Enhanced history management
    if #signature.angle_history > 16 then
        table.remove(signature.angle_history, 1)
    end
    
    -- Enhanced pattern analysis
    if #signature.angle_history >= 8 then
        local rapid_changes = 0
        local pattern_score = 0
        local last_delta = 0
        
        for i = 2, #signature.angle_history do
            local angle_delta = math.abs(
                NormalizeAngle(
                    signature.angle_history[i].yaw - 
                    signature.angle_history[i-1].yaw
                )
            )
            local time_delta = signature.angle_history[i].delta_time
            
            -- Enhanced detection criteria
            if angle_delta > 45 and time_delta < 0.05 then
                rapid_changes = rapid_changes + 1
                
                -- Pattern consistency check
                if last_delta > 0 then
                    local consistency = 1 - math.abs(angle_delta - last_delta) / 180
                    pattern_score = pattern_score + consistency
                end
                
                last_delta = angle_delta
            end
        end
        
        -- Update pattern analysis
        signature.pattern_analysis.consistency_score = 
            pattern_score / (rapid_changes > 0 and rapid_changes or 1)
        
        if rapid_changes >= 4 then
            signature.manipulation_count = signature.manipulation_count + 1
            signature.last_significant_change = current_time
            
            -- Enhanced confidence calculation
            local base_confidence = math.min(
                signature.manipulation_count / 10,
                1.0
            )
            
            -- Adjust confidence based on pattern consistency
            signature.signature_confidence = base_confidence * 
                (0.7 + 0.3 * signature.pattern_analysis.consistency_score)
            
            return true
        end
    end
    
    return false
end

local function ValidateMovement(ent)
    -- Check for defensive/jitter first
    local detection = DetectMetaJitter(ent)
    if detection == "defensive" then
        return "defensive"
    elseif detection then
        return "jitter"
    end
    
    local raw_vel = entity.get_prop(ent, "m_vecVelocity")
    local velocity = (type(raw_vel) == "table") and raw_vel or {0, 0, 0}
    local speed = sqrt(velocity[1]^2 + velocity[2]^2)
    local flags = entity.get_prop(ent, "m_fFlags") or 0
    local is_on_ground = band(flags, 1) ~= 0
    local duck_amount = entity.get_prop(ent, "m_flDuckAmount") or 0
    
    local animstate = SafeAnimstate(ent)
    local anim_speed = animstate and animstate.m_flFeetSpeedForwardsOrSideways or 0
    
    -- Movement state checks
    if speed > 5 and speed < 60 and anim_speed < 0.3 and is_on_ground then
        return "slow-walk"
    end
    
    if not is_on_ground then
        if velocity[3] > 10 then
            return "jump"
        elseif velocity[3] < -10 then
            return "fall"
        end
    end
    
    -- Pattern analysis
    local pattern = AnalyzeMovementPatterns(ent)
    if pattern and pattern.is_strafing then
        return "strafing"
    end
    
    return "normal"
end



local function EnhancedMetaJitterDetection(ent)
    local features = ExtractFeatures(ent)
    local current_time = globals.curtime()
    
    if not anti_aim_signatures[ent] then
        anti_aim_signatures[ent] = CreateAntiAimSignature(ent)
    end
    
    local signature = anti_aim_signatures[ent]
    
    -- Enhanced desync history tracking
    table.insert(signature.desync_history, {
        desync = features.desync,
        time = current_time,
        movement_state = ValidateMovement(ent)
    })
    
    if #signature.desync_history > 32 then
        table.remove(signature.desync_history, 1)
    end
    
    -- Enhanced pattern analysis
    local low_desync_count = 0
    local high_desync_count = 0
    local pattern_consistency = 0
    
    for i = 2, #signature.desync_history do
        local curr = signature.desync_history[i]
        local prev = signature.desync_history[i-1]
        
        if math.abs(curr.desync) < 10 then
            low_desync_count = low_desync_count + 1
        elseif math.abs(curr.desync) > 45 then
            high_desync_count = high_desync_count + 1
        end
        
        -- Calculate pattern consistency
        if i > 2 then
            local delta_curr = math.abs(curr.desync - prev.desync)
            local delta_prev = math.abs(prev.desync - signature.desync_history[i-2].desync)
            pattern_consistency = pattern_consistency + (1 - math.abs(delta_curr - delta_prev) / 90)
        end
    end
    
    pattern_consistency = pattern_consistency / (#signature.desync_history - 2)
    
    -- Enhanced meta jitter detection
    local is_meta_jitter = (
        low_desync_count > #signature.desync_history * 0.7 and 
        high_desync_count < 2 and
        pattern_consistency > 0.6
    )
    
    if is_meta_jitter then
        signature.signature_confidence = math.min(
            signature.signature_confidence + 0.15,
            1.0
        )
        return true
    end
    
    return false
end




local function HandleMetaJitterDesync(ent, velocity, features)
    local meta_jitter_info = meta_jitter_players[ent]
    if not meta_jitter_info then return nil end
    
    -- Enhanced meta jitter analysis
    if math.abs(features.desync) > 15 then
        -- Additional validation before removing
        local pattern = AnalyzeMovementPatterns(ent)
        if pattern.movement_consistency > 0.7 then
            -- Keep tracking if movement is consistent
            meta_jitter_info.confidence = meta_jitter_info.confidence * 0.8
        else
            meta_jitter_players[ent] = nil
            return nil
        end
    end
    
    -- Enhanced velocity-based adjustment
    local speed = sqrt((velocity[1] or 0)^2 + (velocity[2] or 0)^2)
    local duck_amount = entity.get_prop(ent, "m_flDuckAmount") or 0
    
    if speed > 20 then
        -- Dynamic desync calculation based on speed and duck state
        local base_desync = 5 * meta_jitter_info.side
        local speed_factor = Clamp(speed / 250, 0, 1)
        local duck_factor = 1 - (duck_amount * 0.3)
        
        return Clamp(base_desync * speed_factor * duck_factor, -5, 5)
    end
    
    return nil
end

local function PredictWithEnhancedNN(ent)
    local features = ExtractEnhancedFeatures(ent)
    local anim_state = AnalyzeAnimationLayers(ent)
    
    -- Combine animation analysis with features
    features.pattern_strength = anim_state.pattern_strength
    features.layer_confidence = anim_state.confidence
    features.max_desync_detected = anim_state.is_max_desync and 1 or 0
    
    -- Process data for neural network
    local processed_data = ProcessNeuralNetworkData(features)
    
    -- Get prediction from enhanced neural network
    local predictions, hidden1, hidden2 = EnhancedForwardPass(processed_data)
    
    -- Calculate confidence metrics
    local confidence = {
        eye_yaw = math.min(abs(predictions[1] - 0.5) * 4, 1.0), -- Doubled multiplier
        desync = math.min(abs(predictions[2] - 0.5) * 4, 1.0),  -- Doubled multiplier
        pitch = math.min(abs(predictions[3] - 0.5) * 4, 1.0)    -- Doubled multiplier
    }
    
    -- Store prediction results for further analysis
    if not NN.prediction_history then NN.prediction_history = {} end
    if not NN.prediction_history[ent] then NN.prediction_history[ent] = {} end
    
    table.insert(NN.prediction_history[ent], {
        predictions = predictions,
        confidence = confidence,
        time = globals.curtime(),
        features = features
    })
    
    -- Keep history manageable
    if #NN.prediction_history[ent] > 32 then
        table.remove(NN.prediction_history[ent], 1)
    end
    if predictions[2] < 0.1 then
        predictions[2] = 0.5  -- Default to center (0 desync after transformation)
    end
    
    return predictions, confidence, {hidden1 = hidden1, hidden2 = hidden2}
end


local function IntegrateAnimationAnalysis(ent, current_desync)
    local anim_state = AnalyzeAnimationLayers(ent)
    local nn_prediction, confidence = PredictWithEnhancedNN(ent)
    
    if confidence.desync > 0.8 and anim_state.confidence > 0.7 then
        if anim_state.is_max_desync then
            return 58 * (current_desync > 0 and 1 or -1)
        else
            return current_desync * 0.5
        end
    end
    
    -- Use neural network prediction if confident
    if confidence.desync > 0.9 then
        return (nn_prediction[2] * 116 - 58)
    end
    
    return current_desync
end

local function AdvancedAntiAimDetection(ent)
    local detection_methods = {
        DetectRapidAngleManipulation(ent),
        EnhancedMetaJitterDetection(ent)
    }
    
    -- Neural network prediction for validation
    local nn_prediction, confidence = PredictWithEnhancedNN(ent)
    
    -- Enhanced detection scoring
    local detected_count = 0
    local total_confidence = 0
    
    for _, detected in ipairs(detection_methods) do
        if detected then
            detected_count = detected_count + 1
        end
    end
    
    -- Integrate neural network confidence
    local final_confidence = (detected_count / #detection_methods) * 0.7 +
                           (confidence.desync * 0.3)
    
    return detected_count > 0, final_confidence
end

local function ModifyResolverBasedOnDetection(ent)
    local is_suspicious, confidence = AdvancedAntiAimDetection(ent)
    local features = ExtractEnhancedFeatures(ent)
    
    if is_suspicious then
        -- Enhanced resolver strategy selection
        if confidence > 0.7 then
            -- Store successful detection patterns
            if not NN.detection_patterns then NN.detection_patterns = {} end
            if not NN.detection_patterns[ent] then NN.detection_patterns[ent] = {} end
            
            table.insert(NN.detection_patterns[ent], {
                confidence = confidence,
                features = features,
                time = globals.curtime()
            })
            
            -- Keep pattern history manageable
            if #NN.detection_patterns[ent] > 32 then
                table.remove(NN.detection_patterns[ent], 1)
            end
            
            return true, confidence
        elseif confidence > 0.4 then
            -- Medium confidence handling
            return true, 0.5
        end
    end
    
    return false, 0
end


-- Enhanced Neural Network Prediction Integration


local function BuildAimMatrix(ent)
    local matrix = {
        head = { angle = 0, weight = 0.6 },
        chest = { angle = 0, weight = 0.3 },
        legs = { angle = 0, weight = 0.1 }
    }

    local eye_yaw = entity.get_prop(ent, "m_angEyeAngles[1]") or 0
    local body_yaw = (entity.get_prop(ent, "m_flPoseParameter", POSE_PARAMS.BODY_YAW) or 0) * 116 - 58
    local velocity = PredictAdjustedVelocity(ent)

    matrix.head.angle = eye_yaw + (body_yaw * 0.3)
    matrix.chest.angle = body_yaw + (eye_yaw * 0.15)
    matrix.legs.angle = deg(math.atan2(velocity[2], velocity[1]))

    return matrix
end


local function CalculateEyeYaw(ent, features, aim_matrix)
    local base_angle = aim_matrix.head.angle
    local features = ExtractEnhancedFeatures(ent)
    -- Combine both approaches based on speed and other factors
    if features.speed > 1 then
        -- Moving: Use a dynamic blend between head and legs angles
        local speed_factor = Clamp(features.speed / 250, 0, 1)
        base_angle = base_angle * (1 - speed_factor * 0.3) + 
                     aim_matrix.legs.angle * (speed_factor * 0.3)
    else
        -- Standing: Use a more aggressive approach
        base_angle = base_angle * 1.1 + aim_matrix.chest.angle * 0.2
    end

    -- Enhanced LBY integration
    if features.speed > 0.1 then
        local lby = entity.get_prop(ent, "m_flLowerBodyYawTarget") or base_angle
        local weight = Clamp(features.speed / 80, 0, 0.65)  -- More gradual LBY integration
        base_angle = base_angle * (1 - weight) + lby * weight
    end

    return base_angle
end
local function CalculateDesync(ent, features, aim_matrix)
    -- Dynamic desync base depending on movement
    local desync_base = 45  -- Middle ground between aggressive (58) and precision (35)
    local features = ExtractEnhancedFeatures(ent)
    -- Adjust based on speed
    if features.speed > 10 then
        local speed_factor = Clamp(features.speed / 250, 0, 1)
        desync_base = desync_base * (1 - speed_factor * 0.4)  -- Gradually reduce desync with speed
    end
    
    -- Enhance stability based on animation state
    if features.playback_rate < 0.1 then
        desync_base = desync_base * 1.2  -- Slightly more aggressive when stable
    end

    local desync_sign = (features.desync > 25) and 1 or -1  -- Adjusted threshold
    return desync_base * desync_sign
end


local function CalculatePitchAverage(ent)
    if not pitch_history[ent] or #pitch_history[ent].angles == 0 then
        return 0
    end
    
    local sum = 0
    for _, data in ipairs(pitch_history[ent].angles) do
        sum = sum + data.pitch
    end
    return sum / #pitch_history[ent].angles
end


local function ResolvePitch(ent)
    local eye_pitch = entity.get_prop(ent, "m_angEyeAngles[0]") or 0
    
    -- Initialize or get pitch history
    if not pitch_history[ent] then
        pitch_history[ent] = {
            angles = {},
            patterns = {},
            confidence = 0
        }
    end
    
    -- Enhanced pitch history tracking
    table.insert(pitch_history[ent].angles, {
        pitch = eye_pitch,
        time = globals.curtime(),
        validated = false
    })
    
    -- Maintain history size
    if #pitch_history[ent].angles > PITCH_HISTORY_SIZE then
        table.remove(pitch_history[ent].angles, 1)
    end
    
    -- Get movement context
    local velocity = {entity.get_prop(ent, "m_vecVelocity")}
    local speed = sqrt(velocity[1]^2 + velocity[2]^2)
    local is_moving = speed > 0.1
    local is_on_ground = band(entity.get_prop(ent, "m_fFlags"), 1) ~= 0
    
    -- Enhanced pattern detection
    local pattern_detected = false
    local pattern_type = "none"
    if #pitch_history[ent].angles > 4 then
        local rapid_changes = 0
        local last_direction = 0
        local direction_switches = 0
        
        for i = 2, #pitch_history[ent].angles do
            local curr = pitch_history[ent].angles[i].pitch
            local prev = pitch_history[ent].angles[i-1].pitch
            local delta = curr - prev
            
            -- Detect rapid changes
            if math.abs(delta) > PITCH_FAKE_FLICK_THRESHOLD then
                rapid_changes = rapid_changes + 1
                
                -- Track direction changes
                local current_direction = delta > 0 and 1 or -1
                if last_direction ~= 0 and current_direction ~= last_direction then
                    direction_switches = direction_switches + 1
                end
                last_direction = current_direction
            end
        end
        
        -- Enhanced pattern classification
        if rapid_changes >= 3 then
            pattern_detected = true
            if direction_switches >= 2 then
                pattern_type = "alternating"
            else
                pattern_type = "static"
            end
        end
        
        -- Store pattern information
        table.insert(pitch_history[ent].patterns, {
            type = pattern_type,
            time = globals.curtime(),
            switches = direction_switches,
            changes = rapid_changes
        })
        
        -- Keep pattern history manageable
        if #pitch_history[ent].patterns > 16 then
            table.remove(pitch_history[ent].patterns, 1)
        end
    end
    
    -- Neural network integration for pitch
    local nn_prediction, confidence = PredictWithEnhancedNN(ent)
    local nn_pitch = (nn_prediction[3] * 178) - 89
    
    -- Calculate final pitch with all factors
    local final_pitch = eye_pitch
    local pitch_confidence = 0
    
    if pattern_detected then
        -- Handle detected patterns
        if pattern_type == "alternating" then
            final_pitch = CalculatePitchAverage(ent)
            pitch_confidence = 0.8
        elseif pattern_type == "static" then
            final_pitch = pitch_history[ent].angles[1].pitch
            pitch_confidence = 0.7
        end
    elseif confidence.pitch > 0.8 then
        -- Use neural network prediction if confident
        final_pitch = nn_pitch
        pitch_confidence = confidence.pitch
    else
        -- Default to movement-based resolution
        if not is_moving and is_on_ground then
            final_pitch = CalculatePitchAverage(ent)
            pitch_confidence = 0.6
        end
    end
    
    -- Update pitch history confidence
    pitch_history[ent].confidence = pitch_confidence
    
    -- Store last pitch
    last_pitch[ent] = final_pitch
    
    return final_pitch, pitch_confidence
end

-- Enhanced bruteforce mechanism
local function ApplyBruteforce(ent)
    if not hit_miss_data[ent] then
        hit_miss_data[ent] = {
            hits = 0,
            misses = 0,
            last_angle = 0,
            patterns = {},
            success_rate = {}
        }
    end

    local data = hit_miss_data[ent]
    local miss_count = data.misses or 0
    
    -- Enhanced angle selection based on historical success
    local angles = {0, 58, -58, 29, -29, 45, -45}
    local best_angle = 0
    local best_success = 0
    
    -- Analyze success rates for each angle
    for angle, success in pairs(data.success_rate) do
        if success > best_success then
            best_success = success
            best_angle = angle
        end
    end
    
    -- Use best performing angle if success rate is significant
    if best_success > 0.6 then
        return best_angle + random(-2, 2)
    end
    
    -- Dynamic angle selection based on miss patterns
    local index = (miss_count % #angles) + 1
    local base_angle = angles[index]
    
    -- Add pattern-based variation
    local pattern_offset = 0
    if #data.patterns > 0 then
        local recent_pattern = data.patterns[#data.patterns]
        if recent_pattern.type == "switching" then
            pattern_offset = recent_pattern.last_offset * -1
        elseif recent_pattern.type == "static" then
            pattern_offset = recent_pattern.last_offset
        end
    end
    
    -- Apply final angle with controlled randomization
    local final_angle = base_angle + pattern_offset + random(-3, 3)
    
    -- Store the used angle for pattern analysis
    table.insert(data.patterns, {
        angle = final_angle,
        time = globals.curtime(),
        type = #data.patterns > 0 and "switching" or "static",
        last_offset = pattern_offset
    })
    
    -- Keep pattern history manageable
    if #data.patterns > 16 then
        table.remove(data.patterns, 1)
    end
    
    return final_angle
end

local function ResolveAdvancedAngles(ent)
    FixAnimationLayers(ent)
    CorrectPoseParameters(ent)
    
    local features = ExtractEnhancedFeatures(ent)
    local move_state = ValidateMovement(ent)
    local nn_prediction, nn_confidence = PredictWithEnhancedNN(ent)
    
    -- Only resolve pitch if defensive
    if move_state == "defensive" then
        local pitch = features.on_ground and ResolvePitch(ent) or 0
        return nil, nil, Clamp(pitch, -89, 89)
    end
    
    local eye_yaw, desync, pitch
    
    -- More conservative neural network prediction use
    if nn_confidence.desync > 0.3 then  -- Higher confidence threshold
        eye_yaw = nn_prediction[1] * 360
        desync = (nn_prediction[2] * 116 - 58)
        pitch = nn_prediction[3] * 178 - 89
    else
        -- Fallback to traditional methods
        local aim_matrix = BuildAimMatrix(ent)
        eye_yaw = CalculateEyeYaw(ent, features, aim_matrix)
        desync = features.desync
        pitch = features.on_ground and ResolvePitch(ent) or 0
    end
    eye_yaw = NormalizeAngle(eye_yaw)
    
    -- Strict pitch clamping
    pitch = Clamp(pitch, -89, 89)

    return eye_yaw, desync, pitch
end 

local function UpdateHitPatterns(ent, was_hit, angle_used)
    if not hit_miss_data[ent] then
        hit_miss_data[ent] = {
            hits = 0,
            misses = 0,
            patterns = {},
            success_rate = {}
        }
    end
    
    -- Update hits/misses counters
    if was_hit then
        hit_miss_data[ent].hits = hit_miss_data[ent].hits + 1
    else
        hit_miss_data[ent].misses = hit_miss_data[ent].misses + 1
    end
    
    -- Add pattern data
    table.insert(hit_miss_data[ent].patterns, {
        angle = angle_used or (plist.get(ent, "Force body yaw value") or 0),
        time = globals.curtime(),
        was_hit = was_hit
    })
    
    -- Keep patterns history manageable
    if #hit_miss_data[ent].patterns > 20 then
        table.remove(hit_miss_data[ent].patterns, 1)
    end
    
    -- Update success rate
    local total = hit_miss_data[ent].hits + hit_miss_data[ent].misses
    hit_miss_data[ent].success_rate[tostring(angle_used)] = was_hit and 1 or 0
end


-- Enhanced Visual Functions
-- Graph drawing utilities
local function DrawGraph(x, y, width, height, data, color_r, color_g, color_b, alpha, label, y_min, y_max)
    -- Background with better opacity
    renderer.rectangle(x, y, width, height, 20, 20, 20, 180)
    renderer.rectangle(x, y, width, height, 40, 40, 40, 80, true)
    
    -- Label with text that works reliably (simpler flags)
    renderer.text(x + 3, y - 15, 255, 255, 255, 255, "b", 0, label)
    
    -- Graph bounds display - simplified text flags
    renderer.text(x - 2, y, 255, 255, 255, 150, "r", 0, string.format("%.1f", y_max))
    renderer.text(x - 2, y + height, 255, 255, 255, 150, "r", 0, string.format("%.1f", y_min))
    
    -- Draw graph line
    if #data > 1 then
        for i = 2, #data do
            local prev_value = data[i-1]
            local curr_value = data[i]
            
            -- Normalize values to graph height
            local prev_y = y + height - ((prev_value - y_min) / (y_max - y_min) * height)
            local curr_y = y + height - ((curr_value - y_min) / (y_max - y_min) * height)
            
            -- Calculate x positions based on number of data points
            local prev_x = x + ((i-2) / (#data-1)) * width
            local curr_x = x + ((i-1) / (#data-1)) * width
            
            -- Draw line segment
            renderer.line(prev_x, prev_y, curr_x, curr_y, color_r, color_g, color_b, alpha)
        end
    end
    
    -- Current value indicator with simpler text
    if #data > 0 then
        local current = data[#data]
        local current_y = y + height - ((current - y_min) / (y_max - y_min) * height)
        renderer.circle(x + width, current_y, color_r, color_g, color_b, alpha, 3, 0, 1)
        renderer.text(x + width + 5, current_y, color_r, color_g, color_b, alpha, "", 0, string.format("%.2f", current))
    end
    
    -- Draw grid lines for better readability
    local grid_count = 3
    for i = 1, grid_count - 1 do
        local grid_y = y + (height / grid_count) * i
        renderer.line(x, grid_y, x + width, grid_y, 100, 100, 100, 40)
    end
end

-- Neural Network Visualization
local function DrawNeuralNetworkGraph(x, y, width, height, ent)
    -- Ensure we have data for this entity
    if not NN.prediction_history or not NN.prediction_history[ent] or #NN.prediction_history[ent] == 0 then
        return
    end
    
    -- Extract confidence values for graph
    local confidence_data = {}
    for _, prediction in ipairs(NN.prediction_history[ent]) do
        if prediction.confidence and prediction.confidence.desync then
            table.insert(confidence_data, prediction.confidence.desync)
        end
    end
    
    -- Extract prediction values for second series
    local prediction_data = {}
    for _, prediction in ipairs(NN.prediction_history[ent]) do
        if prediction.predictions and prediction.predictions[2] then
            -- Convert from [0,1] to [-58,58] range for desync
            local desync_value = (prediction.predictions[2] * 116 - 58)
            table.insert(prediction_data, math.abs(desync_value) / 58) -- Normalize to [0,1]
        end
    end
    
    -- Background for section
    local section_height = height * 2 + 85
    --renderer.rectangle(x - 5, y - 20, width + 15, section_height, 10, 10, 10, 100)
    
    -- Draw confidence graph
    DrawGraph(x, y, width, height, confidence_data, 64, 190, 230, 255, "NN Confidence", 0, 1)
    
    -- Draw prediction graph below
    if #prediction_data > 0 then
        DrawGraph(x, y + height + 20, width, height, prediction_data, 210, 130, 240, 255, "Desync Prediction", 0, 1)
    end
    
    -- Draw accuracy metrics in a better format
    local hits = 0
    local total = 0
    if NN.performance_metrics and NN.performance_metrics[ent] and NN.performance_metrics[ent].hit_history then
        for _, hit in ipairs(NN.performance_metrics[ent].hit_history) do
            total = total + 1
            if hit then hits = hits + 1 end
        end
    end
    
    local accuracy = total > 0 and (hits / total) or 0
    
    -- Stats background
    local stats_y = y + height * 2 + 30
    --renderer.rectangle(x, stats_y, width, 20, 20, 20, 180)
    
    -- Draw statistics with simpler text flags
    local stats_text = string.format("Accuracy: %d%% (%d/%d) | LR: %.3f", math.floor(accuracy * 100), hits, total, NN.learning_rate)

    
    -- Draw both texts separately with simpler flags
    renderer.text(x + 88, stats_y + 10, 255, 255, 255, 255, "c", 0, stats_text)

end

-- Resolver Performance Visualization
local function DrawResolverGraph(x, y, width, height, ent)
    -- Create background for entire section
    local section_height = height + 110
    if hit_miss_data and hit_miss_data[ent] then
        section_height = height + 170
    end
    --renderer.rectangle(x - 5, y - 30, width + 15, section_height, 10, 10, 10, 100)
    
    -- Header with simpler text flags
    renderer.text(x + width/2, y - 20, 210, 190, 230, 255, "c", 0, "Resolver Status")
    
    -- Main resolver box with better opacity
    --renderer.rectangle(x, y, width, 24, 25, 25, 25, 220)
    
    -- Get resolver state
    local resolver_active = plist.get(ent, "Force body yaw")
    local resolver_value = plist.get(ent, "Force body yaw value") or 0
    local move_state = ValidateMovement(ent)
    
    -- Resolver type indicator with color
    local type_color = {r = 255, g = 255, b = 255}
    if move_state == "defensive" then
        type_color = {r = 255, g = 100, b = 100} -- Red
    elseif move_state == "jitter" then
        type_color = {r = 255, g = 200, b = 0} -- Yellow
    elseif move_state == "slow-walk" then
        type_color = {r = 100, g = 200, b = 255} -- Blue
    elseif move_state == "strafing" then
        type_color = {r = 180, g = 255, b = 140} -- Green
    end
    
    -- Draw resolver status with simpler text flags
    renderer.text(x + 10, y + 12, type_color.r, type_color.g, type_color.b, 255, "", 0, 
        string.format("Type: %s", move_state))
    
    -- Draw resolver value with gauge
    local value_percent = (resolver_value + 58) / 116 -- Normalize from [-58,58] to [0,1]
    local bar_width = width * 0.5 -- Bar takes 50% of box width
    local bar_x = x + width - bar_width - 10
    
    -- Bar background with better visibility
    renderer.rectangle(bar_x, y + 7, bar_width, 10, 40, 40, 40, 220)
    
    -- Bar fill - gradient from left (red) to center (green) to right (blue)
    local fill_width = bar_width * value_percent
    
    -- Color based on position (red to green to blue)
    local fill_r, fill_g, fill_b
    if value_percent < 0.5 then
        -- Red to green gradient (left half)
        local blend = value_percent * 2 -- 0 to 1 in left half
        fill_r = 255 - (blend * 205)
        fill_g = 50 + (blend * 205)
        fill_b = 50
    else
        -- Green to blue gradient (right half)
        local blend = (value_percent - 0.5) * 2 -- 0 to 1 in right half
        fill_r = 50
        fill_g = 255 - (blend * 105)
        fill_b = 50 + (blend * 205)
    end
    
    -- Draw fill bar
    renderer.rectangle(bar_x, y + 7, fill_width, 10, fill_r, fill_g, fill_b, 230)
    
    -- Add numerical display of resolver value with simpler text flag
    renderer.text(bar_x + bar_width + 10, y + 12, 255, 255, 255, 255, "", 0, 
        string.format("%d°", math.floor(resolver_value)))
    
    -- Draw historical hit/miss data if available
    local hit_data = hit_miss_data[ent]
    if hit_data then
        -- Create hit/miss history graph data
        local history_data = {}
        local success_rate = {}
        
        -- Fill with placeholder if no pattern data exists
        if not hit_data.patterns or #hit_data.patterns == 0 then
            for i = 1, 10 do
                table.insert(history_data, 0)
                table.insert(success_rate, 0)
            end
        else
            -- Calculate hit success over time
            local hits = hit_data.hits or 0
            local misses = hit_data.misses or 0
            local total = hits + misses
            
            for i, pattern in ipairs(hit_data.patterns) do
                -- Use pattern angles as data points, normalized to [-1,1] range
                table.insert(history_data, pattern.angle / 58)
                
                -- Calculate success rate at each point
                local rate = total > 0 and (hits / total) or 0
                table.insert(success_rate, rate)
            end
        end
        
        -- Draw angle history graph
        DrawGraph(x, y + 40, width, height - 40, history_data, 150, 220, 255, 220, "Angle History", -1, 1)
        
        -- Draw success rate below
        DrawGraph(x, y + height + 15, width, 50, success_rate, 100, 255, 100, 220, "Hit Rate", 0, 1)
    end
end

-- Simplified Debug Visualization
local function EnhancedDebugVisuals()
    if not ui.get(resolver_debug) then return end
    
    local screen_width, screen_height = client.screen_size()
    local graph_width = 180  -- Consistent width
    local graph_height = 80  -- Slightly smaller for better proportions
    local x_position = screen_width - 215    -- More flush with screen edge
    local y_position = math.floor(screen_height / 3.5)  -- Cleaner position
    
    -- Create main background for entire visualization
    local total_height = graph_height * 2 + 230  -- Adjusted for cleaner layout
    renderer.rectangle(x_position - 10, y_position - 30, graph_width + 40, total_height, 0, 0, 0, 100)
    renderer.rectangle(x_position - 10, y_position - 30, graph_width + 40, total_height, 40, 40, 40, 20, true)
    
    -- Target selection
    local target = client.current_threat()
    if not target or not entity.is_alive(target) then
        -- Try to find nearest visible enemy
        local players = entity.get_players(true)
        local closest_dist = math.huge
        for _, ent in ipairs(players) do
            if IsValidEntity(ent) and not entity.is_dormant(ent) then
                local player_x, player_y = entity.get_prop(ent, "m_vecOrigin")
                if player_x then
                    local local_x, local_y = entity.get_prop(entity.get_local_player(), "m_vecOrigin")
                    local dist = math.sqrt((player_x - local_x)^2 + (player_y - local_y)^2)
                    if dist < closest_dist then
                        closest_dist = dist
                        target = ent
                    end
                end
            end
        end
    end
    
    if not target or not entity.is_alive(target) then
        -- Draw inactive status if no target found
        renderer.rectangle(x_position, y_position, graph_width, 30, 20, 20, 20, 180)
        renderer.text(x_position + graph_width/2, y_position + 15, 180, 180, 180, 220, "c", 0, "No resolver target")
        return
    end
    
    -- Title with player name - simpler text flags
    local player_name = entity.get_player_name(target)
    renderer.text(x_position + graph_width/2, y_position - 5, 210, 190, 230, 255, "c", 0, "Resolving: " .. player_name)
    
    -- Draw resolver graph - direct position
    DrawResolverGraph(x_position, y_position, graph_width, graph_height, target)
    
    -- Draw neural network graph - positioned below with proper spacing
    DrawNeuralNetworkGraph(x_position, y_position + graph_height + 90, graph_width, graph_height, target)
end

-- Performance monitoring and statistics
local function UpdatePerformanceStats(ent, prediction_error, was_hit)
    if not NN.performance_metrics[ent] then
        NN.performance_metrics[ent] = {
            prediction_errors = {},
            hit_history = {},  -- Initialize this
            last_update = globals.curtime(),
            last_confidence = 0,
            predictions = {},  -- Also initialize predictions
            total_error = 0,
            hits = 0,
            misses = 0
        }
    end
    
    local metrics = NN.performance_metrics[ent]
    
    -- Ensure hit_history exists
    if not metrics.hit_history then
        metrics.hit_history = {}
    end
    
    -- Update prediction errors
    table.insert(metrics.prediction_errors, prediction_error)
    if #metrics.prediction_errors > 32 then
        table.remove(metrics.prediction_errors, 1)
    end
    
    -- Update hit history
    table.insert(metrics.hit_history, was_hit)
    if #metrics.hit_history > 32 then
        table.remove(metrics.hit_history, 1)
    end
    
    -- Calculate recent accuracy
    local recent_hits = 0
    for _, hit in ipairs(metrics.hit_history) do
        if hit then recent_hits = recent_hits + 1 end
    end
    
    -- Calculate average prediction error
    local avg_error = 0
    for _, error in ipairs(metrics.prediction_errors) do
        avg_error = avg_error + error
    end
    avg_error = avg_error / #metrics.prediction_errors
    
    -- Update confidence based on performance
    metrics.last_confidence = (recent_hits / #metrics.hit_history) * 0.6 +
                            (1 - math.min(avg_error / 90, 1)) * 0.4
    
    return metrics.last_confidence
end
-- Part 10: Event Callbacks and Main Integration
-- Neural Network-Enhanced Angle Resolver



local function DebugNeuralNetwork()

    
    local function LogWeightStats(weights, name)
        local min_weight, max_weight = math.huge, -math.huge
        local total_weight = 0
        local weight_count = 0
        
        for _, layer in pairs(weights) do
            if type(layer) == "table" then
                for _, row in pairs(layer) do
                    if type(row) == "number" then
                        -- If row is directly a number
                        min_weight = math.min(min_weight, row)
                        max_weight = math.max(max_weight, row)
                        total_weight = total_weight + row
                        weight_count = weight_count + 1
                    elseif type(row) == "table" then
                        -- If row is a table of weights
                        for _, weight in pairs(row) do
                            if type(weight) == "number" then
                                min_weight = math.min(min_weight, weight)
                                max_weight = math.max(max_weight, weight)
                                total_weight = total_weight + weight
                                weight_count = weight_count + 1
                            end
                        end
                    end
                end
            end
        end
        
        -- Avoid division by zero
        local avg_weight = weight_count > 0 and (total_weight / weight_count) or 0
    end
    LogWeightStats(NN.weights.hidden, "Hidden Layer 1")
    LogWeightStats(NN.weights.hidden2, "Hidden Layer 2")
    LogWeightStats(NN.weights.output, "Output Layer")
end
local debug_info = {
    last_check = 0,
    nn_values = {},
    hit_data = {}
}

-- Add this to an existing callback or create a new one
client.set_event_callback("paint", function()
    if ui.get(resolver_debug) then
        DebugNeuralNetwork()
        if globals.curtime() - debug_info.last_check > 1.0 then
            debug_info.last_check = globals.curtime()
            
            local target = client.current_threat()
            if target and entity.is_alive(target) then
                -- Check NN data
                debug_info.nn_values = {}
                if NN.prediction_history and NN.prediction_history[target] then
                    for i, data in ipairs(NN.prediction_history[target]) do
                        if data.predictions then
                            table.insert(debug_info.nn_values, {
                                raw = data.predictions[2],
                                transformed = (data.predictions[2] * 116 - 58),
                                normalized = math.abs((data.predictions[2] * 116 - 58)) / 58
                            })
                        end
                    end
                end
                
                -- Check hit data
                debug_info.hit_data = {}
                if hit_miss_data[target] then
                    debug_info.hit_data = {
                        hits = hit_miss_data[target].hits or 0,
                        misses = hit_miss_data[target].misses or 0,
                        patterns = #(hit_miss_data[target].patterns or {})
                    }
                end
                

            end
        end
    end
end)


-- Round end cleanup and data persistence
client.set_event_callback("round_end", function()
    -- Clear temporary data
    meta_jitter_players = {}
    anti_aim_signatures = {}
    
    -- Save neural network state if performance is good
    for ent, metrics in pairs(NN.performance_metrics) do
        if metrics.last_confidence > 0.7 then
            -- Save successful training data
            local training_data = {
                weights = NN.weights,
                performance = metrics,
                timestamp = globals.curtime()
            }
            writefile("resolver_data_" .. ent .. ".txt", json.stringify(training_data))
        end
    end
end)


client.set_event_callback("player_hurt", function(e)
    local ent = client.userid_to_entindex(e.userid)
    if ent and IsValidEntity(ent) then
        -- Force multiple training iterations
        for i = 1, 5 do
            local features = ExtractEnhancedFeatures(ent)
            UpdateNeuralNetworkModel(ent, true, features, {
                eye_yaw = entity.get_prop(ent, "m_angEyeAngles[1]"),
                desync = features.desync,
                pitch = entity.get_prop(ent, "m_angEyeAngles[0]")
            })
        end
    end
end)


-- Enhanced hit registration and neural network updates
client.set_event_callback("player_hurt", function(e)
    local ent = client.userid_to_entindex(e.userid)
    if ent and IsValidEntity(ent) then
        local resolver_value = plist.get(ent, "Force body yaw value") or 0
        UpdateHitPatterns(ent, true, resolver_value)
        -- Get current resolution data
        local features = ExtractEnhancedFeatures(ent)
        
        -- Get current angles
        local current_eye_yaw = entity.get_prop(ent, "m_angEyeAngles[1]") or 0
        local current_pitch = entity.get_prop(ent, "m_angEyeAngles[0]") or 0
        local current_desync = features.desync
        
        -- Update neural network with successful hit data
        local current_data = {
            features = features,
            angles = {
                eye_yaw = current_eye_yaw,
                desync = current_desync,
                pitch = current_pitch
            },
            time = globals.curtime()
        }
        
        -- Update performance metrics
        local confidence = UpdatePerformanceStats(ent, 0, true)
        hit_miss_data[ent] = hit_miss_data[ent] or {hits = 0, misses = 0}
        hit_miss_data[ent].hits = hit_miss_data[ent].hits + 1
        
        -- Quality-based data storage
        local quality_score = AssessDataQuality(current_data)
        if quality_score > 0.7 then
            table.insert(ResolverData.successful_hits, current_data)
            ResolverData.hit_counter = ResolverData.hit_counter + 1
            
            -- Save data periodically
            if ResolverData.hit_counter >= ResolverData.save_interval then
                writefile("resolver_data.txt", json.stringify(ResolverData.successful_hits))
                ResolverData.hit_counter = 0
            end
        end
    end
end)

-- Miss handling and model adjustment
client.set_event_callback("weapon_fire", function(e)
    client.delay_call(0.15, function()
        local ent = client.userid_to_entindex(e.userid)
        if ent and IsValidEntity(ent) then
            local resolver_value = plist.get(ent, "Force body yaw value") or 0
            UpdateHitPatterns(ent, false, resolver_value)
            local features = ExtractEnhancedFeatures(ent)
            local current_eye_yaw = entity.get_prop(ent, "m_angEyeAngles[1]") or 0
            
            -- Update neural network with miss data
            UpdateNeuralNetworkModel(ent, false, features, {
                eye_yaw = current_eye_yaw,
                desync = features.desync,
                pitch = entity.get_prop(ent, "m_angEyeAngles[0]") or 0
            })
            
            -- Update miss statistics
            hit_miss_data[ent] = hit_miss_data[ent] or {hits = 0, misses = 0}
            hit_miss_data[ent].misses = math.min(hit_miss_data[ent].misses + 1, 8)
            
            -- Update performance metrics
            UpdatePerformanceStats(ent, 1, false)
        end
    end)
end)

-- Replace the original debug text function with the enhanced visuals
client.set_event_callback("paint", function()
    if not ui.get(resolver_master) or not ui.get(resolver_debug) then return end
    
    -- Call the new enhanced visuals instead of the old debug text
    EnhancedDebugVisuals()
    
    -- Process resolver for all valid players
    local players = entity.get_players(true)
    for _, ent in ipairs(players) do
        if IsValidEntity(ent) and not entity.is_dormant(ent) then
            ResolveAdvancedAngles(ent)
        end
    end
end)
-- Main resolver update
-- Main resolver update
client.set_event_callback("net_update_start", function()
    if not ui.get(resolver_master) then
        -- Reset all player settings
        local players = entity.get_players(true)
        for _, ent in ipairs(players) do
            plist.set(ent, "Force body yaw", false)
        end
        return
    end
    
    local players = entity.get_players(true)
    for _, ent in ipairs(players) do
        if IsValidEntity(ent) and not entity.is_dormant(ent) then
            local resolved_eye_yaw, resolved_desync, resolved_pitch = ResolveAdvancedAngles(ent)
            
            -- Only apply resolver if not defensive
            if resolved_eye_yaw ~= nil then
                plist.set(ent, "Force body yaw", true)
                plist.set(ent, "Force body yaw value", resolved_desync)
                entity.set_prop(ent, "m_angEyeAngles[1]", resolved_eye_yaw)
            else
                plist.set(ent, "Force body yaw", false)
            end
            
            -- Always apply pitch resolution
            entity.set_prop(ent, "m_angEyeAngles[0]", resolved_pitch)
        end
    end
end)


-- Cleanup on shutdown
client.set_event_callback("shutdown", function()
    -- Save final neural network state
    local final_state = {
        weights = NN.weights,
        performance_metrics = NN.performance_metrics,
        timestamp = globals.curtime()
    }
    writefile("resolver_final_state.txt", json.stringify(final_state))
    
    -- Reset player settings
    for i = 1, 64 do
        plist.set(i, "Force body yaw", false)
        plist.set(i, "Force body yaw value", 0)
        plist.set(i, "Force pitch", false)
        
        -- Clear stored data
        resolver_states[i] = nil
        hit_miss_data[i] = nil
        last_pitch[i] = nil
        pitch_history[i] = nil
        previous_yaw[i] = nil
    end
    
    Records = CreateWeakCache()
    resolver_states = CreateWeakCache()
    hit_miss_data = CreateWeakCache()
    last_pitch = CreateWeakCache()
    pitch_history = CreateWeakCache()
    previous_yaw = CreateWeakCache()
end)
