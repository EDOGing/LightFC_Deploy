package com.example.trackingapp.model

import org.json.JSONObject
import java.io.File

data class ModelInfo(
    val id: String,
    val name: String,
    val source: String,
    val templateParamFileName: String,
    val templateBinFileName: String,
    val trackingParamFileName: String,
    val trackingBinFileName: String,
    val templateParamSize: Long,
    val templateBinSize: Long,
    val trackingParamSize: Long,
    val trackingBinSize: Long,
    val importedAtMillis: Long,
    val directory: File,
) {
    val isBundled: Boolean
        get() = source == SOURCE_BUNDLED

    val totalSize: Long
        get() = templateParamSize + templateBinSize + trackingParamSize + trackingBinSize

    val templateParamFile: File
        get() = File(directory, templateParamFileName)
    val templateBinFile: File
        get() = File(directory, templateBinFileName)
    val trackingParamFile: File
        get() = File(directory, trackingParamFileName)
    val trackingBinFile: File
        get() = File(directory, trackingBinFileName)

    fun toJson(): JSONObject = JSONObject().apply {
        put("schema_version", SCHEMA_VERSION)
        put("id", id)
        put("name", name)
        put("source", source)
        put("template_param", templateParamFileName)
        put("template_bin", templateBinFileName)
        put("tracking_param", trackingParamFileName)
        put("tracking_bin", trackingBinFileName)
        put("template_param_size", templateParamSize)
        put("template_bin_size", templateBinSize)
        put("tracking_param_size", trackingParamSize)
        put("tracking_bin_size", trackingBinSize)
        put("imported_at_millis", importedAtMillis)
    }

    companion object {
        const val SCHEMA_VERSION = 2
        const val METADATA_FILE_NAME = "model.json"
        const val SOURCE_BUNDLED = "bundled"
        const val SOURCE_IMPORTED = "imported"

        fun fromDirectory(directory: File): ModelInfo {
            val json = JSONObject(
                File(directory, METADATA_FILE_NAME).readText(Charsets.UTF_8),
            )
            require(json.getInt("schema_version") == SCHEMA_VERSION) { "不支持的模型元数据版本" }
            val id = json.getString("id")
            require(id == directory.name) { "模型目录与元数据 ID 不一致" }

            val info = ModelInfo(
                id = id,
                name = json.getString("name"),
                source = json.getString("source"),
                templateParamFileName = json.getString("template_param"),
                templateBinFileName = json.getString("template_bin"),
                trackingParamFileName = json.getString("tracking_param"),
                trackingBinFileName = json.getString("tracking_bin"),
                templateParamSize = json.getLong("template_param_size"),
                templateBinSize = json.getLong("template_bin_size"),
                trackingParamSize = json.getLong("tracking_param_size"),
                trackingBinSize = json.getLong("tracking_bin_size"),
                importedAtMillis = json.getLong("imported_at_millis"),
                directory = directory,
            )
            require(info.source == SOURCE_BUNDLED || info.source == SOURCE_IMPORTED) {
                "未知模型来源"
            }
            info.requireValidFile(info.templateParamFile, info.templateParamSize, "template param")
            info.requireValidFile(info.templateBinFile, info.templateBinSize, "template bin")
            info.requireValidFile(info.trackingParamFile, info.trackingParamSize, "tracking param")
            info.requireValidFile(info.trackingBinFile, info.trackingBinSize, "tracking bin")
            return info
        }

        private fun ModelInfo.requireValidFile(file: File, expectedSize: Long, label: String) {
            require(file.name == file.path.substringAfterLast(File.separatorChar)) { "$label 文件名非法" }
            require(file.canonicalFile.parentFile == directory.canonicalFile) { "$label 文件路径非法" }
            require(file.isFile && file.length() > 0L && file.length() == expectedSize) {
                "$label 文件缺失、为空或大小不匹配"
            }
        }
    }
}
